import json
import time
import datetime
import numpy as np
#np.set_printoptions(threshold=np.inf)
from collections import defaultdict
import glob
import cv2
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)

from Motion import MotionDetector
from model import CustomModel

# DPT imports
import torch
import torch.nn.functional as F
import util.io
from torchvision.transforms import Compose
from dpt.models import DPTSegmentationModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

# Classifier imports
from tensorflow.keras.applications import * #Efficient Net included here
from tensorflow.keras import models
from tensorflow.keras import layers


from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_prototypes',20,'')
flags.DEFINE_integer('top_k',3,'')
flags.DEFINE_string('model_weights','','')
flags.DEFINE_string('video','','')

flags.DEFINE_float('gamma',300,'')
flags.DEFINE_float('all_margin',0.25,'')

height = 224
width = 224

pred_thresh = 2
diff_thresh = 3
sum_thresh = 1
cond_thresh = 2

mot_frame_buffer = 6
mot_thresh = 20

def run(input_path, output_path, model_path, no_spill_frame, is_image=False, model_type="dpt_hybrid", optimize=True):
    """Run segmentation network

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """

    spill_classifier = CustomModel()
    spill_classifier.build(input_shape=(None,224,224,3))
    print(spill_classifier.prototype_layer.prototypes)

    spill_classifier.load_weights("effnet_weights/"+FLAGS.model_weights+'.h5')
    
    print(spill_classifier.prototype_layer.prototypes)
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    net_w = net_h = 480

    #candidate_spill_categories = torch.tensor([22,28,29,44,61,105,110,121,129,138,142,143,148,23]).reshape(1,-1,1)
    candidate_spill_categories = torch.tensor([22,29,44,61,105,110,121,129,138,142,143,148,23]).reshape(1,-1,1).to('cuda')
    floor_cat = torch.tensor([4]).reshape(1,1,1).to('cuda')
    # water, mirror, rug, sign, river, fountain, swimming_pool, food, lake, tray, screen, plate, glass, painting
    top_k = 3

    model = DPTSegmentationModel(
        150,
        path=model_path,
        backbone="vitb_rn50_384",
    )

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ]
    )

    model.eval()
    model.to(device)

    if not is_image:
        cap = cv2.VideoCapture(input_path)
        ret, frame = cap.read()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter('output_videos/'+FLAGS.model_weights+FLAGS.video+'.avi', fourcc, 5.0, (frame.shape[1],frame.shape[0]))

        motion_det = MotionDetector(frame,mot_frame_buffer,mot_thresh)

        count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            if count % 7 > 0:
                continue

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            img_input = transform({"image": img})["image"]

            # compute
            with torch.no_grad():
                sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

                out = model.forward(sample)

                prediction = torch.nn.functional.interpolate(
                    out, size=img.shape[:2], mode="bicubic", align_corners=False
                )
                max_pred = torch.argmax(prediction, dim=1, keepdim=True) + 1
                max_pred = max_pred.squeeze().cpu().numpy()
                
                sorted_k = torch.argsort(prediction, dim=1, descending=True)[:,:top_k] + 1

                spill_pix = (sorted_k.reshape(top_k,1,-1) == candidate_spill_categories).any(dim=1).any(dim=0)
                floor_pix = (sorted_k[:,:1].reshape(1,-1) == 4).any(dim=0)
                candidate_pix = torch.logical_and(spill_pix,floor_pix).reshape(img.shape[:2])
                candidate_pix = candidate_pix.long()
                seg_mask = candidate_pix.cpu().numpy() + 2
                candidate_pix = candidate_pix.cpu().numpy() * 255

            # output
            #filename = os.path.join(
            #    output_path, os.path.splitext(os.path.basename(img_name))[0]
            #)
            #util.io.write_segm_img(filename, img, seg_mask, alpha=0.5)
            
            img = frame
            #motion_bboxes = motion_det.detect(frame)
            motion_bboxes = []
            crops, no_spill_crops, bboxes = get_spill_crops(frame.copy(), no_spill_frame.copy(), candidate_pix.astype(np.uint8), motion_bboxes)
            for crop,ns_crop,bbox in zip(crops,no_spill_crops,bboxes):
                resized = crop/255.
                resized = tf.image.central_crop(resized,central_fraction=min(resized.shape[0]/resized.shape[1],resized.shape[1]/resized.shape[0]))
                resized = tf.image.resize(resized, [int(resized.shape[0]*3),int(resized.shape[0]*3)])
                resized = tf.image.resize_with_crop_or_pad(resized, target_height=224, target_width=224)
                #resized = tf.image.resize(resized, [224,224])
                resized = tf.reshape(resized,(1,224,224,3))
                outp = spill_classifier(resized,training=False)

                resized = ns_crop/255.
                resized = tf.image.central_crop(resized,central_fraction=min(resized.shape[0]/resized.shape[1],resized.shape[1]/resized.shape[0]))
                resized = tf.image.resize(resized, [int(resized.shape[0]*3),int(resized.shape[0]*3)])
                resized = tf.image.resize_with_crop_or_pad(resized, target_height=224, target_width=224)
                #resized = tf.image.resize(resized, [224,224])
                resized = tf.reshape(resized,(1,224,224,3))
                ns_outp = spill_classifier(resized,training=False)

                embds = tf.concat([outp,ns_outp], axis=0)
                embds = tf.nn.l2_normalize(embds, axis=1)
                #sims_all = tf.matmul(embds,embds,transpose_b=True)
                prototypes = tf.nn.l2_normalize(spill_classifier.prototypes, axis=1)
                sims_all = tf.matmul(embds,prototypes)
                sims_top_k,nn_idxs = tf.math.top_k(sims_all,k=1,sorted=False)

                #conf = outp[:,spill].numpy()[0]
                #pred = outp[0,0]
                #diff = pred-ns_outp[0,0]
                #logits_str = "Pred_{:.1f}_Diff_{:.1f}_Sum_{:.1f}".format(pred,diff,pred+diff)
                
                logits_str = "SimP_{:.2f}_{}_SimN_{:.2f}_{}".format(sims_top_k[0,0].numpy(),nn_idxs[0,0].numpy(),sims_top_k[1,0].numpy(),nn_idxs[1,0].numpy())

                if False:
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
                    cv2.putText(img, logits_str, (bbox[0],bbox[1]-3), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
                    #cv2.putText(img, "Spill", (bbox[0],bbox[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
                else:
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
                    cv2.putText(img, logits_str, (bbox[0],bbox[1]-3), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
                    #cv2.putText(img, "No Spill", (bbox[0],bbox[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)


            for bbox in motion_bboxes:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 1)
            
            writer.write(img)
            cv2.imshow('a',img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    writer.release()
    cv2.destroyAllWindows()
    print("finished")

def get_spill_crops(image, no_spill_frame, spill_seg, motion_bboxes):
    CROP_DELTA = 20
    max_size = 60

    cnts,_ = cv2.findContours(spill_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    img_h,img_w,_ = image.shape
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w < 30 or h < 30:
            continue

        if w > max_size:
            x = x+(w-max_size)//2
            w = max_size
        if h > max_size:
            y = y+(h-max_size)//2
            h = max_size

        crop_bbox = (max(x-CROP_DELTA,0),max(y-CROP_DELTA,0),min(x+w+CROP_DELTA,img_w-1),min(y+h+CROP_DELTA,img_h-1))

        mot_in_crop = False
        for mot_bbox in motion_bboxes:
            x_left = max(mot_bbox[0], crop_bbox[0])
            y_top = max(mot_bbox[1], crop_bbox[1])
            x_right = min(mot_bbox[2], crop_bbox[2])
            y_bottom = min(mot_bbox[3], crop_bbox[3])

            if not (x_right <= x_left or y_bottom <= y_top):
                mot_in_crop = True
                break

        if not mot_in_crop:
            bboxes.append(crop_bbox)
            '''cr_w,cr_h = crop_bbox[2]-crop_bbox[0], crop_bbox[3]-crop_bbox[1]
            bboxes.append((crop_bbox[0],crop_bbox[1],crop_bbox[2]-cr_w//2,crop_bbox[3]-cr_h//2))
            bboxes.append((crop_bbox[0]+cr_w//2,crop_bbox[1],crop_bbox[2],crop_bbox[3]-cr_h//2))
            bboxes.append((crop_bbox[0],crop_bbox[1]+cr_h//2,crop_bbox[2]-cr_w//2,crop_bbox[3]))
            bboxes.append((crop_bbox[0]+cr_w//2,crop_bbox[1]+cr_h//2,crop_bbox[2],crop_bbox[3]))'''
    
    '''overlap = True
    while overlap:
        if len(bboxes) < 2:
            overlap = False
            continue
        overlap = False
        bb_ix = -1
        while (bb_ix := bb_ix+1) < len(bboxes)-1:
            bb1 = bboxes[bb_ix]
            bb_ix2 = bb_ix
            while (bb_ix2 := bb_ix2+1) < len(bboxes):
                bb2 = bboxes[bb_ix2]

                x_left = max(bb1[0], bb2[0])
                y_top = max(bb1[1], bb2[1])
                x_right = min(bb1[2], bb2[2])
                y_bottom = min(bb1[3], bb2[3])

                if x_right <= x_left or y_bottom <= y_top:
                    continue
                else:
                    new_bbox = (min(bb1[0],bb2[0]),min(bb1[1],bb2[1]),max(bb1[2],bb2[2]),max(bb1[3],bb2[3]))
                    cr_w,cr_h = new_bbox[2]-new_bbox[0], new_bbox[3]-new_bbox[1]
                    if cr_w <= max_size and cr_h <= max_size:
                        bboxes[bb_ix] = new_bbox
                        bb_ix = -1
                        del bboxes[bb_ix2]
                        overlap = True
                        break'''

    ret_bboxes = []    
    crops = []
    no_spill_crops = []
    for bbox in bboxes:
        cr_w,cr_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        crops.append(image[bbox[1]:bbox[3],bbox[0]:bbox[2]])
        no_spill_crops.append(no_spill_frame[bbox[1]:bbox[3],bbox[0]:bbox[2]])
        ret_bboxes.append(bbox)
    
    return crops, no_spill_crops, ret_bboxes


def main(argv):

    no_spill_frame = cv2.imread('reference_frames/{}.png'.format(FLAGS.video[:-4]))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    run(
        "input_videos/{}".format(FLAGS.video),
        "output_semseg",
        "weights/dpt_hybrid-ade20k-53898607.pt",
        no_spill_frame
    )

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)
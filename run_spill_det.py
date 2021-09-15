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
from PIL import Image
import torch

from Motion import MotionDetector

from spill_model import SpillDetector
import clip

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('clip_model','ViT-B/32','')
flags.DEFINE_integer('proj_head',0,'')
flags.DEFINE_integer('num_prototypes',20,'')

flags.DEFINE_string('model_weights','','')
flags.DEFINE_string('video','','')

image_size = 720
crop_size = 224

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

    device = 'cuda'

    clip_model, preprocess = clip.load(FLAGS.clip_model,device=device)

    spill_det = SpillDetector(clip_model,device=device)
    spill_det.to(device)
    spill_det.load_state_dict(torch.load('weights/'+FLAGS.model_weights,map_location=torch.device(device)))
    

    #crop_dims = [(3,5),(4,7),(5,8),(6,10)]
    crop_dims = [(3,5),(4,7)]
    #num_patches = [15,28,10,15]
    num_patches = [40]

    if not is_image:
        cap = cv2.VideoCapture(input_path)
        ret, frame = cap.read()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter('output_videos/'+FLAGS.model_weights[:-3]+FLAGS.video, fourcc, 5.0, (frame.shape[1],frame.shape[0]))

        motion_det = MotionDetector(frame,mot_frame_buffer,mot_thresh)

        count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            if count % 7 > 0:
                continue

            cv2.imwrite('temp.png',frame)
            img = Image.open('temp.png')
            img_w,img_h = img.size
            
            with torch.no_grad():
                img_patches = []
                bboxes = []
                for num_p,cr_dim in zip(num_patches,crop_dims):
                    num_y,num_x = cr_dim
                    p_w,p_h = img_w//num_x, img_h//num_y
                    #patch_samples = np.random.choice(num_x*num_y,num_p,replace=False)
                    if cr_dim[0] == 4:
                        x_offset = 1
                    elif cr_dim[0] == 5:
                        x_offset = 2
                    elif cr_dim[0] == 6:
                        x_offset = 3
                    else:
                        x_offset = 0

                    spill_patches = []
                    sampled_patches = []
                    p_count = 0
                    for y_ix in range(num_y):
                        for x_ix in range(x_offset,num_x-x_offset):
                            sampled_patches.append(preprocess(img.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1)))))
                            bboxes.append((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1)))

                            p_count += 1
                            if len(spill_patches) == num_p:
                                break
                        if len(spill_patches) == num_p:
                            break

                    img_patches.extend(sampled_patches)

                img_patches = torch.stack(img_patches).to(device)
                sims = spill_det(img_patches)
                max_sim = sims.max()
                max_ix = torch.argmax(sims.max(dim=1)[0])
                bbox = bboxes[max_ix]
            
            #logits_str = "SimCurr_{:.2f}_{}_SimRef_{:.2f}_{}".format(sims_curr[0,0].numpy(),curr_nn_idxs[0,0].numpy(),sims_ref[0,0].numpy(),ref_nn_idxs[0,0].numpy())
            max_sim = max_sim.cpu().numpy()
            logits_str = "{:.2f}".format(max_sim)

            if max_sim > 0.17:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
                #cv2.putText(frame, logits_str, (bbox[0],bbox[1]-3), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
                cv2.putText(frame, "Spill", (bbox[0]+3,bbox[1]+15), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
            elif max_sim > 0.15:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,165,255), 2)
                #cv2.putText(frame, logits_str, (bbox[0],bbox[1]-3), cv2.FONT_HERSHEY_PLAIN, 1, (0,165,255), 2)
                cv2.putText(frame, "Possible Spill", (bbox[0]+3,bbox[1]+15), cv2.FONT_HERSHEY_PLAIN, 1, (0,165,255), 2)
            else:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
                #cv2.putText(frame, logits_str, (bbox[0],bbox[1]-3), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
                cv2.putText(frame, "No Spill", (bbox[0]+3,bbox[1]+15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)


            #for bbox in motion_bboxes:
            #    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 1)
            
            writer.write(frame)
            cv2.imshow('a',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
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

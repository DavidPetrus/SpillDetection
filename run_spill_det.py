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

#from Motion import MotionDetector

from spill_model import SpillDetector
import clip

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('clip_model','ViT-B/16','')
flags.DEFINE_integer('num_prototypes',30,'')

flags.DEFINE_string('model_weights','','')
flags.DEFINE_float('potential_thresh',0.11,'')
flags.DEFINE_float('spill_thresh',0.13,'')
flags.DEFINE_string('video','','')


# 22September2.pt
# Pool: 0.13, 0.17
# Store: 0.11, 0.13

#mot_frame_buffer = 6
#mot_thresh = 20

def run(input_path):

    device = 'cuda'

    clip_model, preprocess = clip.load(FLAGS.clip_model,device=device)

    spill_det = SpillDetector(clip_model,device=device)
    spill_det.to(device)
    spill_det.prototypes = torch.load('weights/'+FLAGS.model_weights,map_location=torch.device(device))['prototypes']

    skip_frames = 7

    frame_scale = 1

    #crop_dims = [(3,5),(4,7),(5,8),(6,10)]
    crop_dims = [(3,5),(4,7)]
    #crop_dims = [(2,1),(3,1)]
    #crop_dims = [(4,7)]

    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if frame_scale > 1:
        writer = cv2.VideoWriter('output_videos/'+FLAGS.video[:-4]+'_'+FLAGS.model_weights[:-3]+'.avi', fourcc, 24/skip_frames, (540,960))
    else:
        writer = cv2.VideoWriter('output_videos/'+FLAGS.video[:-4]+'_'+FLAGS.model_weights[:-3]+'.avi', fourcc, 24/skip_frames, (frame.shape[1],frame.shape[0]))

    #motion_det = MotionDetector(frame,mot_frame_buffer,mot_thresh)

    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % skip_frames > 0 or count < 2640:
            continue

        cv2.imwrite('temp.png',frame)
        img = Image.open('temp.png')
        img_w,img_h = img.size
        
        with torch.no_grad():
            img_patches = []
            bboxes = []
            for cr_dim in crop_dims:
                if img_w > img_h:
                    num_y = min(cr_dim)
                    num_x = max(cr_dim)
                else:
                    num_y = max(cr_dim)
                    num_x = min(cr_dim)
                p_w,p_h = img_w//num_x, img_h//num_y

                spill_patches = []
                sampled_patches = []

                for y_ix in range(num_y):
                    for x_ix in range(num_x):
                        sampled_patches.append(preprocess(img.crop((p_w*x_ix,p_h*y_ix,p_w*x_ix+p_w,p_h*y_ix+p_h))))
                        bboxes.append((p_w*x_ix,p_h*y_ix,p_w*x_ix+p_w,p_h*y_ix+p_h))

                for y_ix in range(num_y-1):
                    for x_ix in range(num_x-1):
                        sampled_patches.append(preprocess(img.crop((p_w//2+p_w*x_ix,p_h//2+p_h*y_ix,p_w//2+p_w*x_ix+p_w,p_h//2+p_h*y_ix+p_h))))
                        bboxes.append((p_w//2+p_w*x_ix,p_h//2+p_h*y_ix,p_w//2+p_w*x_ix+p_w,p_h//2+p_h*y_ix+p_h))

                img_patches.extend(sampled_patches)

            img_patches = torch.stack(img_patches).to(device)
            sims = spill_det(img_patches)

            max_sims,max_idxs = torch.topk(sims.max(dim=1)[0], 3, sorted=True)

            max_sims = max_sims.cpu().numpy()
            max_idxs = max_idxs.cpu().numpy()

        for max_sim,max_ix in zip(max_sims,max_idxs):
            logits_str = "{:.3f}".format(max_sim)
            bbox = bboxes[max_ix]

            if max_sim > FLAGS.spill_thresh:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2*frame_scale)
                cv2.putText(frame, logits_str, (bbox[0]+3,bbox[1]+15*frame_scale), cv2.FONT_HERSHEY_PLAIN, frame_scale, (0,0,255), 2*frame_scale)
                #cv2.putText(frame, "Spill", (bbox[0]+3,bbox[1]+15*frame_scale), cv2.FONT_HERSHEY_PLAIN, frame_scale, (0,0,255), 2*frame_scale)
            elif max_sim > FLAGS.potential_thresh:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,165,255), 2*frame_scale)
                cv2.putText(frame, logits_str, (bbox[0]+3,bbox[1]+15*frame_scale), cv2.FONT_HERSHEY_PLAIN, frame_scale, (0,165,255), 2*frame_scale)
                #cv2.putText(frame, "Possible Spill", (bbox[0]+3,bbox[1]+15*frame_scale), cv2.FONT_HERSHEY_PLAIN, frame_scale, (0,165,255), 2*frame_scale)
            elif max_sim > FLAGS.potential_thresh - 0.02:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2*frame_scale)
                cv2.putText(frame, logits_str, (bbox[0]+3,bbox[1]+15*frame_scale), cv2.FONT_HERSHEY_PLAIN, frame_scale, (0,255,0), 2*frame_scale)
                #cv2.putText(frame, "No Spill", (bbox[0]+3,bbox[1]+15*frame_scale), cv2.FONT_HERSHEY_PLAIN, frame_scale, (0,255,0), 2*frame_scale)


        #for bbox in motion_bboxes:
        #    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 1)

        if frame_scale > 1:
            frame = cv2.resize(frame, (540,960)) 
        
        writer.write(frame)
        cv2.imshow('a',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    cv2.destroyAllWindows()
    print("finished")

def main(argv):

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    run(
        "input_videos/{}".format(FLAGS.video)
    )

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)

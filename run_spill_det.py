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
import torchvision
import torchvision.transforms.functional as F_vis

from Motion import MotionDetector

from spill_model import SpillDetector
import clip

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('clip_model','ViT-B/16','')
flags.DEFINE_integer('proj_hidden',0,'')
flags.DEFINE_integer('proj_head',128,'')
flags.DEFINE_integer('num_prototypes',30,'')
flags.DEFINE_integer('num_proto_sets',1,'')

flags.DEFINE_string('model_weights','','')
flags.DEFINE_float('potential_thresh',0.2,'')
flags.DEFINE_float('spill_thresh',0.38,'')
flags.DEFINE_string('video','','')
flags.DEFINE_string('camera','','')

flags.DEFINE_float('frame_avg_coeff',1.,'')


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
    state_dict = torch.load('weights/'+FLAGS.model_weights,map_location=torch.device(device))
    spill_det.prototypes = state_dict['prototypes']
    if FLAGS.proj_head > 0:
        spill_det.proj_head = state_dict['proj_head']

    skip_frames = 3

    frame_scale = 1

    #x_crops = [(0.,0.2),(0.1,0.3),(0.2,0.4),(0.3,0.5),(0.4,0.6),(0.5,0.7),(0.6,0.8),(0.7,0.9),(0.8,1.)]
    #y_crops = [(0.,0.33),(0.17,0.5),(0.33,0.66),(0.5,0.83),(0.66,1.)]

    x_crops = [(0.,0.281),(0.242,0.523),(0.477,0.758),(0.719,1.)]
    y_crops = [(0.,0.5),(0.25,0.75),(0.5,1.)]

    mot_frame_buffer = 6
    mot_thresh = 20

    #color_transform = torchvision.transforms.Grayscale(num_output_channels=3)
    #color_transform = torchvision.transforms.RandomEqualize(p=1)
    autocontrast = lambda x: F_vis.autocontrast(x)
    #color_transforms = [lambda x: F_vis.adjust_hue(x,0.5), lambda x: F_vis.adjust_hue(x,0.2), lambda x: F_vis.adjust_hue(x,-0.2), lambda x: F_vis.invert(x)]
    #color_transforms = []
    #t_names = ['hue_05','hue_pos','hue_neg','invert']
    #t_names = []

    #vids = glob.glob("/home/petrus/Downloads/SpillData/morningside_clips/*")
    vids = ["/home/petrus/Downloads/SpillData/SpillClips/"+FLAGS.video]

    for vid in vids:
        vid_file = vid.split('/')[-1]
        cap = cv2.VideoCapture(vid)
        ret, frame = cap.read()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if frame_scale > 1:
            writer = cv2.VideoWriter('output_videos/'+vid_file[:-4]+'_'+FLAGS.model_weights[:-3]+'.avi', fourcc, 24/skip_frames, (540,960))
        else:
            writer = cv2.VideoWriter('output_videos/'+vid_file[:-4]+'_'+FLAGS.model_weights[:-3]+'.avi', fourcc, 24/skip_frames, (frame.shape[1],frame.shape[0]))

        motion_det = MotionDetector(frame,mot_frame_buffer,mot_thresh)

        avg_sims = defaultdict(float)
        #floor_pix = np.load('camera_calibration/{}_floor_seg.npy'.format(FLAGS.camera))
        #exc_patches = np.load('camera_calibration/{}.npy'.format(FLAGS.camera)).tolist()

        count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            if count % skip_frames > 0:
                continue

            #frame = color_transform(torch.tensor(frame).movedim(2,0)).movedim(0,2).numpy()

            cv2.imwrite('temp.png',frame)
            frame = cv2.imread('temp.png')
            img = Image.open('temp.png')
            img_w,img_h = img.size

            motion_bboxes = motion_det.detect(frame)

            aug_bboxes = []
            
            with torch.no_grad():
                img_patches = []
                bboxes = []
                for y_dim in y_crops:
                    for x_dim in x_crops:
                        patch_bbox = (int(x_dim[0]*img_w),int(y_dim[0]*img_h),int(x_dim[1]*img_w),int(y_dim[1]*img_h))

                        #if list(patch_bbox) in exc_patches: continue
                        #if floor_pix[patch_bbox[1]:patch_bbox[3],patch_bbox[0]:patch_bbox[2]].mean() < 0.2: continue

                        has_motion = False
                        '''for mot_bbox in motion_bboxes:
                            if get_bbox_overlap(mot_bbox,patch_bbox):
                                has_motion = True
                                break'''

                        if has_motion: continue

                        patch = autocontrast(img.crop(patch_bbox))
                        img_patches.append(preprocess(patch))
                        bboxes.append((patch_bbox,'org'))
                        #for t,n in zip(color_transforms,t_names):
                        #    img_patches.append(preprocess(t(patch)))
                        #    bboxes.append((patch_bbox,n))

                img_patches = torch.stack(img_patches).to(device)
                sims = spill_det(img_patches)

                max_sims,max_idxs = torch.sort(sims.max(dim=1)[0], descending=True)

                max_sims = max_sims[:4].cpu().numpy()
                max_idxs = max_idxs[:4].cpu().numpy()

            frame_bbs = []
            for p_sim,p_idx in zip(max_sims,max_idxs):
                '''if bboxes[p_idx] not in avg_sims and bboxes[p_idx] in aug_bboxes:
                    avg_sims[bboxes[p_idx]] = FLAGS.potential_thresh - 0.01
                else:
                    avg_sims[bboxes[p_idx]] = (1-FLAGS.frame_avg_coeff)*avg_sims[bboxes[p_idx]] + FLAGS.frame_avg_coeff*p_sim'''
                #avg_sims[bboxes[p_idx]] = (1-FLAGS.frame_avg_coeff)*avg_sims[bboxes[p_idx]] + FLAGS.frame_avg_coeff*p_sim
                frame_bbs.append(bboxes[p_idx])

            #index_max = max(avg_sims, key=avg_sims.get)
            #max_sim = avg_sims[index_max]

            #for mot_bbox in motion_bboxes:
            #    cv2.rectangle(frame, (mot_bbox[0], mot_bbox[1]), (mot_bbox[2], mot_bbox[3]), (255,0,0), 1)

            #for bbox_tup,max_sim in avg_sims.items():
            m_count = 0
            for bbox_tup,max_sim in zip(frame_bbs,max_sims):
                bbox,aug = bbox_tup
                logits_str = "{:.3f}".format(max_sim)

                if max_sim > FLAGS.spill_thresh:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2*frame_scale)
                    cv2.putText(frame, logits_str, (bbox[0]+3,bbox[1]+15*frame_scale), cv2.FONT_HERSHEY_PLAIN, frame_scale, (0,0,255), 2*frame_scale)
                    #cv2.putText(frame, "Spill", (bbox[0]+3,bbox[1]+15*frame_scale), cv2.FONT_HERSHEY_PLAIN, frame_scale, (0,0,255), 2*frame_scale)
                elif max_sim > FLAGS.potential_thresh:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,165,255), 2*frame_scale)
                    cv2.putText(frame, logits_str, (bbox[0]+3,bbox[1]+15*frame_scale), cv2.FONT_HERSHEY_PLAIN, frame_scale, (0,165,255), 2*frame_scale)
                    #cv2.putText(frame, "Possible Spill", (bbox[0]+3,bbox[1]+15*frame_scale), cv2.FONT_HERSHEY_PLAIN, frame_scale, (0,165,255), 2*frame_scale)
                    #elif max_sim > FLAGS.potential_thresh-0.02:
                elif m_count == 0:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2*frame_scale)
                    cv2.putText(frame, logits_str, (bbox[0]+3,bbox[1]+15*frame_scale), cv2.FONT_HERSHEY_PLAIN, frame_scale, (0,255,0), 2*frame_scale)
                    #cv2.putText(frame, "No Spill", (bbox[0]+3,bbox[1]+15*frame_scale), cv2.FONT_HERSHEY_PLAIN, frame_scale, (0,255,0), 2*frame_scale)

                m_count += 1

            if frame_scale > 1:
                frame = cv2.resize(frame, (540,960)) 
            
            writer.write(frame)
            cv2.imshow('a',frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        writer.release()
        cv2.destroyAllWindows()
        print("finished")

def get_bbox_overlap(bb1,bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return False
    else:
        return True

def main(argv):

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    run(
        "input_videos/{}".format(FLAGS.video)
    )

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)

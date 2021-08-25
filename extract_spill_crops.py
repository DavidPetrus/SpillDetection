import numpy as np
import cv2
import glob
import os
from collections import defaultdict

video_dir = "/home/petrus/Downloads/GallonSmashingVideos/gallon1_"

clip_files = glob.glob(video_dir+"*.txt")

min_size = 112

for clip_file in clip_files:
    with open(clip_file,'r') as fp:
        annots = fp.read().splitlines()

    spill_annots = defaultdict(list)
    fuzzy_annots = defaultdict(list)
    unique_bboxes = []
    first_annot = None
    skip_frames = []
    no_spill_frames = []
    for annot in annots:
        if len(annot.split(' '))==2:
            fr_ix,lab = annot.split(' ')
            if lab=='no_spill':
                no_spill_frames.append(int(fr_ix[2:]))
            else:
                skip_frames.append(int(fr_ix[2:]))
            continue
        fr_ix,lab,lux,luy,rux,ruy = annot.split(' ')
        if lab == 'spill':
            fr_ix = int(fr_ix[2:])
            if first_annot is None:
                first_annot = fr_ix
            
            bbox = (int(lux),int(luy),int(rux),int(ruy))
            spill_annots[fr_ix].append(bbox)
            if bbox not in unique_bboxes:
                unique_bboxes.append(bbox)

    cap = cv2.VideoCapture(clip_file[:-4]+'.mp4')
    if not os.path.exists(clip_file[:-4]+'_not_spill'):
        os.makedirs(clip_file[:-4]+'_not_spill')

    if not os.path.exists(clip_file[:-4]):
        os.makedirs(clip_file[:-4])

    spill_crops = []
    count = -1
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break

        frame = frame[100:]
        img_h,img_w,_ = frame.shape

        count += 1
        if count % 5 > 0:
            continue

        if count in no_spill_frames:
            for bb_c,bbox in enumerate(unique_bboxes):
                w,h = bbox[2]-bbox[0],bbox[3]-bbox[1]
                c_x,c_y = int(bbox[0] + w/2),int(bbox[1] + h/2)
                r_w,r_h = max(56,w),max(56,h)
                if c_x+r_w > img_w: c_x = img_w-r_w
                if c_y+r_h > img_h: c_y = img_h-r_h
                if c_x-r_w < 0: c_x = r_w
                if c_y-r_h < 0: c_y = r_h

                spill_crops.append(frame[c_y-r_h:c_y+r_h,c_x-r_w:c_x+r_w])
                cv2.imwrite(clip_file[:-4]+'_not_spill'+'/{}_{}.png'.format(count,bb_c),spill_crops[-1])

        for bb_c,spill_bbox in enumerate(spill_annots[count]):
            w,h = spill_bbox[2]-spill_bbox[0],spill_bbox[3]-spill_bbox[1]
            c_x,c_y = int(spill_bbox[0] + w/2),int(spill_bbox[1] + h/2)
            r_w,r_h = max(56,w),max(56,h)
            if c_x+r_w > img_w: c_x = img_w-r_w
            if c_y+r_h > img_h: c_y = img_h-r_h
            if c_x-r_w < 0: c_x = r_w
            if c_y-r_h < 0: c_y = r_h

            spill_crops.append(frame[c_y-r_h:c_y+r_h,c_x-r_w:c_x+r_w])
            cv2.imwrite(clip_file[:-4]+'/{}_{}.png'.format(count,bb_c),spill_crops[-1])


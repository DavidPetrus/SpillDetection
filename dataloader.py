import numpy as np
import random
import cv2
import torch
import torchvision
import torch.nn.functional as F
import glob
from collections import defaultdict

from PIL import Image

import warnings
warnings.filterwarnings("ignore")

from absl import flags, app

FLAGS = flags.FLAGS

class CustomDataGen(torch.utils.data.Dataset):
    
    def __init__(self, directory, batch_size, preprocess,input_size=(224, 224, 3),train=True):
        
        self.spill_images = glob.glob(directory+'/spills/*')
        self.not_spill_images = glob.glob(directory+'/not_spills/*')
        self.all_images = self.spill_images + self.not_spill_images
        self.img_cats = [0] * len(self.spill_images) + [1] * len(self.not_spill_images)
        #self.all_images, self.img_cats = shuffle(self.all_images, self.img_cats)

        self.puddle_images = glob.glob(directory+'/puddles/*')
        self.not_puddle_images = glob.glob(directory+'/not_puddles/*')
        self.all_puddles = self.puddle_images + self.not_puddle_images
        self.puddle_cats = [0] * len(self.puddle_images) + [1] * len(self.not_puddle_images)
        #self.all_puddles, self.puddle_cats = shuffle(self.all_puddles, self.puddle_cats)

        self.vids = glob.glob(directory+'/spill_vids/*')
        self.gallon = [vid for vid in self.vids if 'gallon' in vid]
        self.rds = [vid for vid in self.vids if not 'gallon' in vid]

        self.preprocess = preprocess
        self.crop_dims = {'spill':[(1,1),(1,2),(2,2),(3,4)],'not_spill':[(3,5),(4,7),(5,8),(6,10)], \
                          'gallon':[(2,3),(3,5),(4,7),(5,8)],'react':[(2,3),(3,5),(4,6)],'drinks':[(3,5),(4,7),(5,8)], \
                          'store':[(3,5),(4,7),(5,8)],'val_vid':[(3,5),(4,7),(5,8),(6,10)]}
        self.num_patches = {'spill':[1,2,4,3],'not_spill':[15,14,10,11],'gallon':[6,8,8,8],'react':[6,10,14], \
                            'drinks':[10,10,10],'store':[10,10,10],'val_vid':[5,5,10,10]}

        self.vid_frames = {}
        self.pos_frames = {}
        self.neg_frames = {}
        self.vid_cats = {}
        for vid in self.vids:
            spill_frames = glob.glob(vid+'/*')
            not_spill_frames = glob.glob(vid.replace('spill_vids','not_spill_vids')+'/*')
            self.vid_frames[vid] = spill_frames + not_spill_frames
            self.vid_cats[vid] = [0] * len(spill_frames) + [1] * len(not_spill_frames)
            self.pos_frames[vid] = spill_frames
            self.neg_frames[vid] = not_spill_frames

        self.batch_size = batch_size
        self.num_crops = {}

        self.zeros = torch.zeros(1024)
        self.ones = torch.ones(1024)

        self.random_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)

        self.num_spill_samples = 30

        self.videos = glob.glob(directory+'/spill_vids/*.mp4') + glob.glob(directory+'/spill_vids/*.avi')

        self.vid_resize_range = {480: (1,2), 608: (0.6,1.5), 720: (0.5,1.5), 1080: (0.6,1.5)}

        self.video_groups = ['gallon','smacking_drinks','react','store1_1','store1_2','store1_3']
        self.vid_probs = [0.6,0.7,0.9,0.93,0.96,1.]
        self.vid_pos = {}
        self.vid_neg = {}
        for gr in self.video_groups:
            self.vid_pos[gr] = glob.glob(directory+'/spill_vids/{}*'.format(gr))
            self.vid_neg[gr] = glob.glob(directory+'/not_spill_vids/{}*'.format(gr))
        
        self.train = train
        if self.train:
            self.n = len(self.spill_images)
        else:
            self.batch_size = batch_size//2
            self.n = len(self.spill_images)
        

    def __get_image__(self, index, dataset='', sampled_frames=None):
        if dataset == 'video':
            vids,frames,bboxes,cats,all_bboxes = sampled_frames
            all_images = frames
        elif dataset == 'spill':
            pos_img = Image.open(self.spill_images[np.random.randint(len(self.spill_images))])
            neg_img = Image.open(self.not_spill_images[np.random.randint(len(self.not_spill_images))])
            neg_img2 = Image.open(self.not_spill_images[np.random.randint(len(self.not_spill_images))])
            all_images = [pos_img,neg_img,neg_img2]
        elif dataset == 'puddle':
            pos_img = Image.open(self.puddle_images[np.random.randint(len(self.puddle_images))])
            neg_img = Image.open(self.not_puddle_images[np.random.randint(len(self.not_puddle_images))])
            neg_img2 = Image.open(self.not_puddle_images[np.random.randint(len(self.not_puddle_images))])
            all_images = [pos_img,neg_img,neg_img2]

        imgs = []
        for i_ix,img in enumerate(all_images):
            cat = 0 if i_ix==0 else 1

            img_w,img_h = img.size
            img_size = min(img_h,img_w)

            spill_mask = np.zeros([img_h,img_w,1],dtype=np.float32)

            if self.train:
                if dataset == 'spill':
                    cr_h,cr_w = np.random.randint(int(0.7*img_h),img_h+1),np.random.randint(int(0.7*img_w),img_w+1)
                    cr_y,cr_x = np.random.randint(0,img_h-cr_h+1), np.random.randint(0,img_w-cr_w+1)
                    crop = img.crop((cr_x,cr_y,cr_x+cr_w,cr_y+cr_h))

                    crop_dims = self.crop_dims['spill'] if cat == 0 else self.crop_dims['not_spill']
                    num_patches = self.num_patches['spill'] if cat == 0 else self.num_patches['not_spill']
                    all_patches = []
                    for num_p,cr_dim in zip(num_patches,crop_dims):
                        min_dim,max_dim = cr_dim
                        num_x = min_dim if cr_h > cr_w else max_dim
                        num_y = max_dim if cr_h > cr_w else min_dim
                        p_w,p_h = cr_w//num_x, cr_h//num_y
                        patch_samples = np.random.choice(num_x*num_y,num_p,replace=False)
                        p_count = 0
                        for x_ix in range(num_x):
                            for y_ix in range(num_y):
                                if p_count in patch_samples:
                                    all_patches.append(self.random_flip(self.preprocess(crop.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1))))))
                                p_count += 1

                elif dataset == 'video':
                    frame_bboxes = all_bboxes[i_ix]
                    for bb in frame_bboxes:
                        spill_mask[max(bb[1],0):min(bb[3],img_h),max(bb[0],0):min(bb[2],img_w)] = 1.

                    cat = cats[i_ix]
                    bbox = bboxes[i_ix]
                    lux,luy,rbx,rby = bbox
                    bb_w = rbx-lux
                    bb_h = rby-luy
                    cr_l = np.random.randint(0,lux+1)
                    cr_t = np.random.randint(0,luy+1)
                    cr_r = np.random.randint(rbx,img_w)
                    cr_b = np.random.randint(rby,img_h)
                    
                    crop_size = min(cr_r-cr_l,cr_b-cr_t)
                    if crop_size < bb_w:
                        crop_size = cr_r-cr_l
                        if luy > img_h-1-rby:
                            cr_t = max(cr_b-crop_size,0)
                        else:
                            cr_b = min(cr_t+crop_size,img_h-1)
                        crop_size = min(cr_r-cr_l,cr_b-cr_t)
                    elif crop_size < bb_h:
                        crop_size = cr_b-cr_t
                        if lux > img_w-1-rbx:
                            cr_l = max(cr_r-crop_size,0)
                        else:
                            cr_r = min(cr_l+crop_size,img_w-1)
                        crop_size = min(cr_r-cr_l,cr_b-cr_t)

                    if cr_r-cr_l > crop_size:
                        if lux < img_w-1-rbx:
                            cr_l = np.random.randint(max(0,rbx-crop_size),min(lux+1,img_w-crop_size))
                            cr_r = cr_l+crop_size
                        else:
                            cr_r = np.random.randint(max(rbx,crop_size),min(img_w,lux+crop_size+1))
                            cr_l = cr_r-crop_size
                    elif cr_b-cr_t > crop_size:
                        if luy < img_h-1-rby:
                            cr_t = np.random.randint(max(0,rby-crop_size),min(luy+1,img_h-crop_size))
                            cr_b = cr_t+crop_size
                        else:
                            cr_b = np.random.randint(max(rby,crop_size),min(img_h,luy+crop_size+1))
                            cr_t = cr_b-crop_size

                    crop_dim = np.array([cr_l,cr_t,cr_r,cr_b])
                    spill_mask = spill_mask[crop_dim[1]:crop_dim[3],crop_dim[0]:crop_dim[2]]

                    crop = img.crop((cr_l,cr_t,cr_r,cr_b))
                    cr_w,cr_h = cr_r-cr_l, cr_b-cr_t

                    vid_group = vids[0] if i_ix<4 else vids[1]
                    crop_dims = self.crop_dims[vid_group]
                    num_patches = self.num_patches[vid_group]
                    all_patches = []
                    for num_p,cr_dim in zip(num_patches,crop_dims):
                        min_dim,max_dim = cr_dim
                        num_x = min_dim if cr_h > cr_w else max_dim
                        num_y = max_dim if cr_h > cr_w else min_dim
                        p_w,p_h = cr_w//num_x, cr_h//num_y
                        patch_samples = np.random.choice(num_x*num_y,num_p,replace=False)
                        spill_patches = []
                        sampled_patches = []
                        p_count = 0
                        for y_ix in range(num_y):
                            for x_ix in range(num_x):
                                if cat==0 and spill_mask[p_h*y_ix:p_h*(y_ix+1),p_w*x_ix:p_w*(x_ix+1)].mean() > 0.:
                                    spill_patches.append(self.random_flip(self.preprocess(crop.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1))))))
                                elif p_count in patch_samples:
                                    sampled_patches.append(self.random_flip(self.preprocess(crop.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1))))))

                                p_count += 1
                                if len(spill_patches) == num_p:
                                    break
                            if len(spill_patches) == num_p:
                                break

                        all_patches.extend(spill_patches + sampled_patches[:num_p-len(spill_patches)])

                crops = torch.stack(all_patches)
                #crops = self.color_aug(crops)
            else:        
                if dataset == 'spill':
                    crop_dims = self.crop_dims['spill'] if cat == 0 else self.crop_dims['not_spill']
                    num_patches = self.num_patches['spill'] if cat == 0 else self.num_patches['not_spill']
                    all_patches = []
                    for num_p,cr_dim in zip(num_patches,crop_dims):
                        min_dim,max_dim = cr_dim
                        num_x = min_dim if img_h > img_w else max_dim
                        num_y = max_dim if img_h > img_w else min_dim
                        p_w,p_h = img_w//num_x, img_h//num_y
                        patch_samples = np.random.choice(num_x*num_y,num_p,replace=False)
                        p_count = 0
                        for x_ix in range(num_x):
                            for y_ix in range(num_y):
                                if p_count in patch_samples:
                                    all_patches.append(self.preprocess(img.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1)))))
                                p_count += 1
                elif dataset == 'video':
                    cat = cats[i_ix]
                    frame_bboxes = all_bboxes[i_ix]
                    for bb in frame_bboxes:
                        spill_mask[max(bb[1],0):min(bb[3],img_h),max(bb[0],0):min(bb[2],img_w)] = 1.

                    crop_dims = self.crop_dims['val_vid']
                    num_patches = self.num_patches['val_vid']
                    all_patches = []
                    for num_p,cr_dim in zip(num_patches,crop_dims):
                        min_dim,max_dim = cr_dim
                        num_x = min_dim if img_h > img_w else max_dim
                        num_y = max_dim if img_h > img_w else min_dim
                        p_w,p_h = img_w//num_x, img_h//num_y
                        patch_samples = np.random.choice(num_x*num_y,num_p,replace=False)
                        spill_patches = []
                        sampled_patches = []
                        p_count = 0
                        for y_ix in range(num_y):
                            for x_ix in range(num_x):
                                if cat==0 and spill_mask[p_h*y_ix:p_h*(y_ix+1),p_w*x_ix:p_w*(x_ix+1)].mean() > 0.:
                                    spill_patches.append(self.preprocess(img.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1)))))
                                elif p_count in patch_samples:
                                    sampled_patches.append(self.preprocess(img.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1)))))

                                p_count += 1
                                if len(spill_patches) == num_p:
                                    break
                            if len(spill_patches) == num_p:
                                break

                        all_patches.extend(spill_patches + sampled_patches[:num_p-len(spill_patches)])

                crops = torch.stack(all_patches)

            imgs.append(crops)

        return imgs

    #def on_epoch_end(self):
    #    self.all_images, self.img_cats = shuffle(self.all_images, self.img_cats)
    
    def __getitem__(self, index):
        cats = []
        frames = []
        bboxes = []
        all_bboxes = []
        vids = []
        for v in range(2):
            vid = self.videos[np.random.randint(len(self.videos))]
            for vid_group in ['gallon','drinks','react','store']:
                if vid_group in vid:
                    vids.append(vid_group)
                    break

            with open(vid[:-4]+'.txt','r') as fp:
                lines = fp.read().splitlines()

            no_spill_frames = []
            spill_frames = defaultdict(list)
            for line in lines:
                if 'no_spill' in line:
                    fr,lab = line.split(' ')
                    no_spill_frames.append(int(fr[2:]))
                elif 'spill' in line:
                    fr,lab,lux,luy,rbx,rby = line.split(' ')
                    spill_frames[int(fr[2:])].append((int(lux),int(luy),int(rbx),int(rby)))

            neg_fr_idxs = random.sample(no_spill_frames,2)
            pos_fr_idxs = random.sample(list(spill_frames.keys()),2)
            pos_bboxes = []
            for fr in pos_fr_idxs:
                pos_bboxes.extend(spill_frames[fr])

            vid_frames = glob.glob(vid[:-4]+'/*.png')
            for frame_name in vid_frames:
                count = int(frame_name.split('/')[-1][:-4])
                if count in pos_fr_idxs:
                    all_bboxes.append(spill_frames[count])
                    bbox = random.choice(spill_frames[count])
                    cats.append(0)
                elif count in neg_fr_idxs:
                    if len(pos_bboxes)==0:
                        frame = Image.open(frame_name)
                        lux = np.random.randint(frame.size[0]-100)
                        luy = np.random.randint(frame.size[1]-100)
                        s = np.random.randint(100)
                        bbox = (lux,luy,lux+s,luy+s)
                    else:
                        bbox = random.choice(pos_bboxes)
                    all_bboxes.append([])
                    cats.append(1)
                else:
                    continue

                frame = Image.open(frame_name)
                frames.append(frame)
                bboxes.append((max(0,bbox[0]),max(0,bbox[1]),min(frame.size[0]-1,bbox[2]),min(frame.size[1]-1,bbox[3])))

        imgs = self.__get_image__(index, dataset='video', sampled_frames=(vids,frames,bboxes,cats,all_bboxes))
        pos_images = [img for img,cat in zip(imgs,cats) if cat==0]
        neg_images = [img for img,cat in zip(imgs,cats) if cat==1]
        for ix in range(self.batch_size):
            imgs = self.__get_image__(index+ix, dataset='spill')
            pos_images.append(imgs[0])
            neg_images.append(imgs[1])
            neg_images.append(imgs[2])

        X = torch.cat(pos_images+neg_images)
        lab = torch.cat([self.ones[:len(pos_images)], self.zeros[:len(neg_images)]], dim=0)

        return X,lab
    
    def __len__(self):
        return self.n // self.batch_size

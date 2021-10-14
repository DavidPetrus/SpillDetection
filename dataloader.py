import numpy as np
import random
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as F_vis
import glob
from collections import defaultdict

from PIL import Image

import warnings
warnings.filterwarnings("ignore")

from absl import flags, app

FLAGS = flags.FLAGS

class CustomDataGen(torch.utils.data.Dataset):
    
    def __init__(self, directory, batch_size, preprocess,input_size=(224, 224, 3),train=True,color_distorts=None):
        
        if train:
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

            self.all_floors = glob.glob(directory+'/floors/*')
        else:
            self.val_no_spills = glob.glob(directory+"/no_spills/*/*")

            self.val_frames = {}
            for t in ['clear','dark','opaque']:
                self.val_frames[t] = {}

                v_dir = directory+"/{}_spills".format(t)
                vids = glob.glob(v_dir+"/*.txt")

                for v in vids:
                    self.val_frames[t][v[:-4]] = []
                    imgs = glob.glob(v[:-4]+"/*")

                    with open(v,'r') as fp:
                        lines = fp.read().splitlines()

                    no_spill_frames = []
                    spill_frames = {}
                    for line in lines:
                        if ' spill' in line:
                            fr,lab,lux,luy,rbx,rby = line.split(' ')
                            if v[:-4]+"/{}.png".format(fr[2:]) in imgs:
                                self.val_frames[t][v[:-4]].append((v[:-4]+"/{}.png".format(fr[2:]),(int(lux),int(luy),int(rbx),int(rby))))

        self.color_distorts = color_distorts
        #if FLAGS.autocontrast:
        #    self.autocontrast = lambda x: F_vis.autocontrast(x)
        #else:
        self.autocontrast = lambda x: x

        self.preprocess = preprocess
        if FLAGS.scale == 'all':
            self.crop_dims = {'spill':[(1,1),(1,2),(2,2),(2,3)],'not_spill':[(2,3),(3,5),(4,7)], \
                              'puddle':[(1,1),(1,2),(2,2),(3,1)],'not_puddle':[(1,1),(1,2),(2,2),(1,3)], \
                              'gallon':[(1,2),(2,3),(3,5),(4,7)],'react':[(1,2),(2,3),(3,5),(4,6)],'drinks':[(2,3),(3,5),(4,7)],'store':[(2,3),(3,5),(4,7)], \
                              'pool':[(2,1),(3,1)],'val_vid':[(2,3),(3,5),(4,7)]}

            self.num_patches = {'spill':[1,2,4,3],'not_spill':[6,15,9],'puddle':[1,2,4,3],'not_puddle':[1,2,4,3], \
                                'gallon':[2,6,12,10],'react':[2,6,15,7], 'drinks':[6,14,10],'store':[6,14,10], \
                                'pool':[2,3],'val_vid':[6,10,14]}
        elif FLAGS.scale == 'xlarge':
            self.crop_dims = {'spill':[(1,1)],'not_spill':[(2,3),(3,5)], \
                              'puddle':[(1,1)],'not_puddle':[(1,1),(1,2)], \
                              'gallon':[(1,1),(1,2)],'react':[(1,1),(1,2)],'drinks':[(1,1),(2,3)],'store':[(1,1),(2,3)], \
                              'pool':[(2,1),(3,1)]}

            self.num_patches = {'spill':[1],'not_spill':[6,15],'puddle':[1],'not_puddle':[1,2], \
                                'gallon':[1,2],'react':[1,2], 'drinks':[1,2],'store':[1,2], \
                                'pool':[2,3]}
        elif FLAGS.scale == 'large':
            self.crop_dims = {'spill':[(1,1)],'not_spill':[(2,3),(3,5)], \
                              'puddle':[(1,1),(1,2)],'not_puddle':[(1,1),(1,2)], \
                              'gallon':[(1,2),(2,3)],'react':[(1,2),(2,3)],'drinks':[(2,3),(3,5)],'store':[(2,3),(3,5)], \
                              'pool':[(2,1),(3,1)],'val_vid':[(2,3),(3,5),(4,7)]}

            self.num_patches = {'spill':[1],'not_spill':[6,15],'puddle':[1,2],'not_puddle':[1,2], \
                                'gallon':[2,6],'react':[2,6], 'drinks':[6,2],'store':[6,2], \
                                'pool':[2,3],'val_vid':[6,10,14]}
        elif FLAGS.scale == 'med':
            self.crop_dims = {'spill':[(1,1),(1,2)],'not_spill':[(2,3),(3,5),(4,7)], \
                              'puddle':[(1,1),(1,2)],'not_puddle':[(1,1),(1,2)], \
                              'gallon':[(1,2),(2,3)],'react':[(1,2),(2,3)],'drinks':[(2,3),(3,5)],'store':[(2,3),(3,5)], \
                              'pool':[(2,1),(3,1)],'val_vid':[(2,3),(3,5),(4,7)]}

            self.num_patches = {'spill':[1,1],'not_spill':[6,15,9],'puddle':[1,1],'not_puddle':[1,2], \
                                'gallon':[1,2],'react':[1,2], 'drinks':[1,2],'store':[1,2], \
                                'pool':[2,3],'val_vid':[6,10,14]}
        elif FLAGS.scale == 'small':
            self.crop_dims = {'spill':[(1,2),(2,2),(2,3)],'not_spill':[(3,5),(4,7)], \
                              'puddle':[(1,3),(2,2),(3,1)],'not_puddle':[(1,2),(2,2),(1,3)], \
                              'gallon':[(3,5),(4,7)],'react':[(3,5),(4,6)],'drinks':[(3,5),(4,7)],'store':[(3,5),(4,7)], \
                              'pool':[(2,1),(3,1)],'val_vid':[(2,3),(3,5),(4,7)]}

            self.num_patches = {'spill':[2,4,4],'not_spill':[15,15],'puddle':[3,4,3],'not_puddle':[2,4,3], \
                                'gallon':[15,15],'react':[15,15], 'drinks':[15,15],'store':[15,15], \
                                'pool':[2,3],'val_vid':[6,10,14]}

        if not train:
            self.pool_spills = glob.glob(directory+'/pool_spills/*')
            self.pool_no_spills = glob.glob(directory+'/pool_no_spills/*')
            self.water_spills = glob.glob(directory+'/water_spills/*')
            self.other_spills = glob.glob(directory+'/other_spills/*')
            self.store_no_spills = glob.glob(directory+'/store_no_spills/*')

        if train:
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
            self.n = len(self.spill_images)*3
        else:
            #self.val_samples = ['pool','large_water','small_water','large_other','small_other','spill']
            #self.num_vals = [60,30,15,30,15,20]
            self.batch_size = 1
            self.n = 20
        

    def __get_image__(self, index, dataset='', sampled_frames=None):
        if dataset == 'video':
            vids,frames,bboxes,cats,all_bboxes = sampled_frames
            all_images = frames
        elif dataset == 'val_video':
            all_images,bboxes = sampled_frames
        elif dataset == 'spill':
            pos_img = Image.open(self.spill_images[np.random.randint(len(self.spill_images))])
            neg_img = Image.open(self.not_spill_images[np.random.randint(len(self.not_spill_images))])
            neg_img2 = Image.open(self.not_spill_images[np.random.randint(len(self.not_spill_images))])
            all_images = [pos_img,neg_img,neg_img2]
        elif dataset == 'puddle':
            pos_img = Image.open(self.puddle_images[np.random.randint(len(self.puddle_images))])
            neg_img = Image.open(self.not_puddle_images[np.random.randint(len(self.not_puddle_images))])
            all_images = [pos_img,neg_img]
        elif dataset == 'pool':
            pos_img = Image.open(self.pool_spills[np.random.randint(len(self.pool_spills))])
            neg_img = Image.open(self.pool_no_spills[np.random.randint(len(self.pool_no_spills))])
            all_images = [pos_img,neg_img]
        elif 'water' in dataset:
            pos_img = Image.open(self.water_spills[np.random.randint(len(self.water_spills))])
            neg_img = Image.open(self.store_no_spills[np.random.randint(len(self.store_no_spills))])
            all_images = [pos_img,neg_img]
        elif 'other' in dataset:
            pos_img = Image.open(self.other_spills[np.random.randint(len(self.other_spills))])
            neg_img = Image.open(self.store_no_spills[np.random.randint(len(self.store_no_spills))])
            all_images = [pos_img,neg_img]

        imgs = []
        for i_ix,img_load in enumerate(all_images):
            cat = 0 if i_ix==0 else 1

            img_w,img_h = img_load.size

            if self.train:
                floor = random.choice(self.floors)
                cr_size = np.random.randint(int(0.7*min(floor.size[0],floor.size[1])),min(floor.size[0],floor.size[1])+1)
                cr_x,cr_y = np.random.randint(0,floor.size[0]-cr_size+1), np.random.randint(0,floor.size[1]-cr_size+1)
                floor_org = floor.crop((cr_x,cr_y,cr_x+cr_size,cr_y+cr_size))
                if floor_org.mode != 'RGB':
                    floor_org = floor_org.convert('RGB')

            spill_mask = np.zeros([img_h,img_w,1],dtype=np.float32)
            all_patches = []

            if img_load.mode != 'RGB':
                img_load = img_load.convert('RGB')

            for c in self.color_distorts:

                if c is None:
                    img = img_load
                    if self.train: floor = floor_org
                else:
                    img = c(img_load)
                    if self.train: floor = c(floor_org)

                if self.train:
                    if dataset == 'spill':
                        cr_h,cr_w = np.random.randint(int(0.7*img_h),img_h+1),np.random.randint(int(0.7*img_w),img_w+1)
                        cr_y,cr_x = np.random.randint(0,img_h-cr_h+1), np.random.randint(0,img_w-cr_w+1)
                        crop = img.crop((cr_x,cr_y,cr_x+cr_w,cr_y+cr_h))

                        crop_dims = self.crop_dims['spill'] if cat == 0 else self.crop_dims['not_spill']
                        num_patches = self.num_patches['spill'] if cat == 0 else self.num_patches['not_spill']
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
                                        patch = crop.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1)))
                                        if random.random() < FLAGS.superimpose_frac:
                                            p_floor = floor.copy()

                                            resize_frac = np.random.random()*(FLAGS.max_spill_frac-FLAGS.min_spill_frac) + FLAGS.min_spill_frac
                                            if p_w > p_h:
                                                patch = patch.resize((int(resize_frac*floor.size[0]),int(resize_frac*floor.size[0]*p_h/p_w)))
                                            else:
                                                patch = patch.resize((int(resize_frac*floor.size[0]*p_w/p_h),int(resize_frac*floor.size[0])))

                                            patch.putalpha(np.random.randint(FLAGS.min_alpha,FLAGS.max_alpha))
                                            p_floor.paste(patch, (np.random.randint(floor.size[0]-patch.size[0]), np.random.randint(floor.size[1]-patch.size[1]+1)), patch)
                                            all_patches.append(self.random_flip(self.preprocess(self.autocontrast(p_floor))))
                                        else:
                                            all_patches.append(self.random_flip(self.preprocess(self.autocontrast(patch))))

                                    p_count += 1
                    elif dataset == 'puddle':
                        cr_h,cr_w = np.random.randint(int(0.85*img_h),img_h+1),np.random.randint(int(0.85*img_w),img_w+1)
                        cr_y,cr_x = np.random.randint(0,img_h-cr_h+1), np.random.randint(0,img_w-cr_w+1)
                        crop = img.crop((cr_x,cr_y,cr_x+cr_w,cr_y+cr_h))

                        crop_dims = self.crop_dims['puddle'] if cat == 0 else self.crop_dims['not_puddle']
                        num_patches = self.num_patches['puddle'] if cat == 0 else self.num_patches['not_puddle']
                        for num_p,cr_dim in zip(num_patches,crop_dims):
                            num_y,num_x = cr_dim
                            p_w,p_h = cr_w//num_x, cr_h//num_y
                            for x_ix in range(num_x):
                                for y_ix in range(num_y):
                                    all_patches.append(self.random_flip(self.preprocess(self.autocontrast(crop.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1)))))))

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
                        spill_pix_sum = spill_mask.sum()

                        crop = img.crop((cr_l,cr_t,cr_r,cr_b))
                        cr_w,cr_h = cr_r-cr_l, cr_b-cr_t

                        vid_group = vids[0] if i_ix<4 else vids[1]
                        spill_patches = []
                        crop_dims = self.crop_dims[vid_group]
                        num_patches = self.num_patches[vid_group]
                        for num_p,cr_dim in zip(num_patches,crop_dims):
                            min_dim,max_dim = cr_dim
                            num_x = min_dim if cr_h > cr_w else max_dim
                            num_y = max_dim if cr_h > cr_w else min_dim
                            p_w,p_h = cr_w//num_x, cr_h//num_y
                            patch_samples = np.random.choice(num_x*num_y,num_p,replace=False)
                            sampled_patches = []
                            p_count = 0
                            for y_ix in range(num_y):
                                for x_ix in range(num_x):
                                    if cat==0:
                                        patch_spill_sum = spill_mask[p_h*y_ix+max(0,(p_h-p_w)//2):p_h*(y_ix+1)-max(0,(p_h-p_w)//2),p_w*x_ix+max(0,(p_w-p_h)//2):p_w*(x_ix+1)-max(0,(p_w-p_h)//2)].sum()
                                        if patch_spill_sum > 3000 or patch_spill_sum > 0.8*spill_pix_sum:
                                            spill_patches.append(self.random_flip(self.preprocess(self.autocontrast(crop.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1)))))))
                                    elif cat==1 and p_count in patch_samples:
                                        sampled_patches.append(self.random_flip(self.preprocess(self.autocontrast(crop.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1)))))))

                                    p_count += 1

                            if cat==1:
                                all_patches.extend(sampled_patches)

                        if cat==0:
                            if len(spill_patches) == 0:
                                print(spill_mask.sum(), spill_mask.shape)
                            all_patches.append(random.choice(spill_patches))
                else:        
                    if dataset == 'pool':
                        for crop_dim in [(2,1),(3,1)]:
                            num_y,num_x = crop_dim
                            p_w,p_h = img_w//num_x, img_h//num_y
                            for y_ix in range(num_y):
                                for x_ix in range(num_x):
                                    all_patches.append(self.preprocess(self.autocontrast(img.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1))))))
                    else:
                        has_spill = True if i_ix < len(bboxes) else False

                        if has_spill:
                            bb = bboxes[i_ix]
                            spill_mask[max(bb[1],0):min(bb[3],img_h),max(bb[0],0):min(bb[2],img_w)] = 1.
                            for crop_dim in [(2,3),(3,5)]:
                                max_iou = 0.
                                num_y,num_x = crop_dim
                                p_w,p_h = img_w//num_x, img_h//num_y
                                for y_ix in range(num_y):
                                    for x_ix in range(num_x):
                                        patch_bbox = (p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1))
                                        iou = spill_mask[patch_bbox[1]:patch_bbox[3],patch_bbox[0]:patch_bbox[2]].mean()
                                        if iou > max_iou:
                                            max_iou = iou
                                            spill_patch = img.crop(patch_bbox)

                                for y_ix in range(num_y-1):
                                    for x_ix in range(num_x-1):
                                        patch_bbox = (p_w//2+p_w*x_ix,p_h//2+p_h*y_ix,p_w//2+p_w*(x_ix+1),p_h//2+p_h*(y_ix+1))
                                        iou = spill_mask[patch_bbox[1]:patch_bbox[3],patch_bbox[0]:patch_bbox[2]].mean()
                                        if iou > max_iou:
                                            max_iou = iou
                                            spill_patch = img.crop(patch_bbox)
                                
                                all_patches.append(self.preprocess(self.autocontrast(spill_patch)))
                        else:
                            for crop_dim in [(2,3),(3,5)]:
                                num_y,num_x = crop_dim
                                p_w,p_h = img_w//num_x, img_h//num_y
                                for y_ix in range(num_y):
                                    for x_ix in range(num_x):
                                        all_patches.append(self.preprocess(self.autocontrast(img.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1))))))

                                for y_ix in range(num_y-1):
                                    for x_ix in range(num_x-1):
                                        all_patches.append(self.preprocess(self.autocontrast(img.crop((p_w//2+p_w*x_ix,p_h//2+p_h*y_ix,p_w//2+p_w*(x_ix+1),p_h//2+p_h*(y_ix+1))))))

            imgs.append(torch.stack(all_patches))

        return imgs

    #def on_epoch_end(self):
    #    self.all_images, self.img_cats = shuffle(self.all_images, self.img_cats)
    
    def __getitem__(self, index):
        if self.train:
            cats = []
            frames = []
            bboxes = []
            all_bboxes = []
            vids = []
            for v in range(2):
                vid = self.videos[np.random.randint(len(self.videos))]
                for vid_group in ['gallon','drinks','react','store','pool']:
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
        
            floors_sample = random.sample(self.all_floors,3)
            self.floors = [Image.open(floor) for floor in floors_sample]

            imgs = self.__get_image__(index, dataset='video', sampled_frames=(vids,frames,bboxes,cats,all_bboxes))
            pos_images = [img for img,cat in zip(imgs,cats) if cat==0]
            neg_images = [img for img,cat in zip(imgs,cats) if cat==1]
            for ix in range(self.batch_size):
                imgs = self.__get_image__(index+ix, dataset='spill')
                pos_images.append(imgs[0])
                neg_images.append(imgs[1])
                if FLAGS.scale == 'large':
                    neg_images.append(imgs[2])

            for ix in range(self.batch_size//2):
                imgs = self.__get_image__(index+ix, dataset='puddle')
                pos_images.append(imgs[0])
                neg_images.append(imgs[1])

            X = torch.cat(pos_images+neg_images)
            lab = torch.cat([self.ones[:len(pos_images)], self.zeros[:len(neg_images)]], dim=0)

            return X,lab
        else:
            frames = []
            bboxes = []
            for t in ['clear','dark','opaque']:
                for v,img_names in self.val_frames[t].items():
                    fr_sample,bbox = random.sample(img_names,1)[0]
                    frames.append(Image.open(fr_sample))
                    bboxes.append(bbox)

            no_spill_frames = random.sample(self.val_no_spills,FLAGS.val_batch)
            for ns_frame in no_spill_frames:
                frames.append(Image.open(ns_frame))

            imgs = self.__get_image__(index, dataset='val_video', sampled_frames=(frames,bboxes))
            pos_images = imgs[:len(bboxes)]
            neg_images = imgs[len(bboxes):]

            #for ix in range(30):
            #    imgs = self.__get_image__(index, dataset='pool')
            #    pos_images.append(imgs[0])
            #    neg_images.append(imgs[1])

            return torch.cat(pos_images+neg_images)
        
    
    def __len__(self):
        return self.n // self.batch_size

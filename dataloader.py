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
    
    def __init__(self, directory, batch_size, preprocess,input_size=(224, 224, 3),train=True,color_distorts=None,batch_nums=None):
        
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

            self.all_floors = glob.glob(directory+'/floors/*')
        else:
            self.val_frames = {}
            self.val_no_spills = {}
            for t in ['clear','dark','opaque']:
                self.val_frames[t] = {}
                self.val_no_spills[t] = {}

                v_dir = directory+"/{}_spills".format(t)
                vids = glob.glob(v_dir+"/*.txt")

                for v in vids:
                    self.val_frames[t][v[:-4]] = []
                    imgs = glob.glob(v[:-4]+"/*")

                    with open(v,'r') as fp:
                        lines = fp.read().splitlines()

                    for line in lines:
                        if ' spill' in line:
                            fr,lab,lux,luy,rbx,rby = line.split(' ')
                            if v[:-4]+"/{}.png".format(fr[2:]) in imgs:
                                self.val_frames[t][v[:-4]].append((v[:-4]+"/{}.png".format(fr[2:]),(int(lux),int(luy),int(rbx),int(rby))))

                    for cam in glob.glob(directory+"/no_spills/*"):
                        if cam.split('/')[-1] in v:
                            self.val_no_spills[t][v[:-4]] = glob.glob(cam+"/*")
                            break

        self.color_distorts = color_distorts

        self.batch_nums = batch_nums

        self.preprocess = preprocess

        self.crop_dims = {'spill':[(1,1)],'not_spill':[(2,3),(3,5)], \
                          'puddle':[(1,1)],'not_puddle':[(1,1),(1,2)], \
                          'gallon':[(1,1),(2,2)],'react':[(1,1),(2,2)],'drinks':[(1,1),(2,2)],'store':[(1,1),(2,2)], \
                          'pool':[(2,1),(3,1)], 'morningside':[(1,2),(2,2)]}

        self.num_patches = {'spill':[1],'not_spill':[6,15],'puddle':[1],'not_puddle':[1,2], \
                            'gallon':[1,4],'react':[1,4], 'drinks':[1,4],'store':[1,4], \
                            'pool':[2,3], 'morningside':[1,4]}

        if not train:
            self.pool_spills = glob.glob(directory+'/pool_spills/*')
            self.pool_no_spills = glob.glob(directory+'/pool_no_spills/*')
            self.water_spills = glob.glob(directory+'/water_spills/*')
            self.other_spills = glob.glob(directory+'/other_spills/*')
            self.store_no_spills = glob.glob(directory+'/store_no_spills/*')

        if train:
            self.batch_size = batch_size
            self.num_crops = {}

            self.zeros = torch.zeros(1024)
            self.ones = torch.ones(1024)

            self.random_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)

            self.num_spill_samples = 30

            self.videos = glob.glob(directory+'/spill_vids/*.mp4') + glob.glob(directory+'/spill_vids/*.avi')
            self.video_groups = ['gallon','smacking_drinks','react','store1_1','store1_2','store1_3']

            self.morningside = glob.glob(directory+'/spill_vids/morningside/*.txt')
            self.morningside.sort()
            self.num_spills = [16,3,10,10,5,3,4,7,3,6,7,8,7,5,7,6,6,10,7]
            self.yt_vid_num_spills = [3,2,1,2,3,2,1,1,2,1,1,2,2,3,2,1,2,2,1,1,1,1,2,1,1,1,1,2,4,1,1,1,1,1,1,1]
            self.yt_probs = [n/sum(self.yt_vid_num_spills) for n in self.yt_vid_num_spills]
            self.vid_probs = [num_spill/sum(self.num_spills) for num_spill in self.num_spills]
            self.morningside_no_spill = {}
            for v in self.morningside:
                self.morningside_no_spill[v] = glob.glob(directory+'/spill_vids/morningside/no_spill_videos/'+v[38:-10]+'*')
        
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
        elif dataset == 'morningside':
            frames,bboxes,cats = sampled_frames
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
                                            all_patches.append(self.random_flip(self.preprocess(p_floor)))
                                        else:
                                            all_patches.append(self.random_flip(self.preprocess(patch)))

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
                                    all_patches.append(self.random_flip(self.preprocess(crop.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1))))))

                    elif dataset == 'video' or dataset == 'morningside':
                        #print(dataset,len(bboxes),i_ix)
                        frame_bboxes = all_bboxes[i_ix] if dataset=='video' else [bboxes[i_ix]]
                        for bb in frame_bboxes:
                            spill_mask[max(bb[1],0):min(bb[3],img_h),max(bb[0],0):min(bb[2],img_w)] = 1.

                        cat = cats[i_ix]
                        bbox = bboxes[i_ix]
                        lux,luy,rbx,rby = bbox
                        bb_w = rbx-lux
                        bb_h = rby-luy
                        cc = 0
                        while True:
                            if FLAGS.scale == 'full':
                                cr_l = np.random.randint(0,lux+1)
                                cr_t = np.random.randint(0,luy+1)
                                cr_r = rbx + np.random.randint(0,img_w-rbx)
                                cr_b = rby + np.random.randint(0,img_h-rby)
                            elif FLAGS.scale == 'large':
                                cr_l = np.random.randint(0,3*lux//4 + 1)
                                cr_t = np.random.randint(0,3*luy//4 + 1)
                                cr_r = rbx + np.random.randint((img_w-rbx)//4,img_w-rbx + 1)
                                cr_b = rby + np.random.randint((img_h-rby)//4,img_h-rby + 1)
                            elif FLAGS.scale == 'small':
                                cr_l = np.random.randint(lux//2,lux + 1)
                                cr_t = np.random.randint(luy//2,luy + 1)
                                cr_r = rbx + np.random.randint(0,(img_w-rbx)//2 + 1)
                                cr_b = rby + np.random.randint(0,(img_h-rby)//2 + 1)

                            cr_w,cr_h = cr_r-cr_l, cr_b-cr_t
                            if cr_w > cr_h and cr_w < min(img_w,img_h):
                                if cr_t+cr_w < img_h:
                                    cr_b = cr_t+cr_w
                                else:
                                    cr_t = cr_b-cr_w
                            else:
                                if cr_l+cr_h < img_w:
                                    cr_r = cr_l+cr_h
                                else:
                                    cr_l = cr_r-cr_h

                            if cr_t < 0 or cr_l < 0 or cr_b > img_h or cr_r > img_w:
                                cc += 1
                                if cc > 20:
                                    print("!!!!!!!!!!!!!!!!!!!!!!!!!",cc)
                            else:
                                break

                        crop_dim = np.array([cr_l,cr_t,cr_r,cr_b])
                        spill_mask = spill_mask[crop_dim[1]:crop_dim[3],crop_dim[0]:crop_dim[2]]
                        spill_pix_sum = spill_mask.sum()

                        crop = img.crop((cr_l,cr_t,cr_r,cr_b))
                        cr_w,cr_h = cr_r-cr_l, cr_b-cr_t

                        if dataset=='video':
                            vid_group = vids[0] if i_ix<len(all_images)/2 else vids[1]
                        elif dataset=='morningside':
                            vid_group = 'morningside'

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
                                        if patch_spill_sum > 3000 or patch_spill_sum > 0.6*spill_pix_sum:
                                            spill_patches.append(self.random_flip(self.preprocess(crop.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1))))))
                                    elif cat==1 and p_count in patch_samples:
                                        sampled_patches.append(self.random_flip(self.preprocess(crop.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1))))))

                                    p_count += 1

                            if cat==1:
                                all_patches.extend(sampled_patches)

                        if cat==0:
                            if len(spill_patches) == 0:
                                patch_spill_sum = spill_mask[max(0,(cr_h-cr_w)//2):cr_h-max(0,(cr_h-cr_w)//2),max(0,(cr_w-cr_h)//2):cr_w-max(0,(cr_w-cr_h)//2)].sum()
                                #print(spill_mask.sum(), patch_spill_sum, spill_mask.shape)
                                spill_patches.append(self.random_flip(self.preprocess(crop)))

                            all_patches.append(random.choice(spill_patches))
                else:        
                    if dataset == 'pool':
                        for crop_dim in [(2,1),(3,1)]:
                            num_y,num_x = crop_dim
                            p_w,p_h = img_w//num_x, img_h//num_y
                            for y_ix in range(num_y):
                                for x_ix in range(num_x):
                                    all_patches.append(self.preprocess(img.crop((p_w*x_ix,p_h*y_ix,p_w*(x_ix+1),p_h*(y_ix+1)))))
                    else:
                        has_spill = True if i_ix < len(bboxes) else False
                        x_crops = [[(0.,0.281),(0.242,0.523),(0.477,0.758),(0.719,1.)],[(0.,0.2),(0.1,0.3),(0.2,0.4),(0.3,0.5),(0.4,0.6),(0.5,0.7),(0.6,0.8),(0.7,0.9),(0.8,1.)]]
                        y_crops = [[(0.,0.5),(0.25,0.75),(0.5,1.)],[(0.,0.33),(0.17,0.5),(0.33,0.66),(0.5,0.83),(0.66,1.)]]
                        for val_crop_x,val_crop_y in zip(x_crops,y_crops):
                            if has_spill:
                                bb = bboxes[i_ix]
                                spill_mask[max(bb[1],0):min(bb[3],img_h),max(bb[0],0):min(bb[2],img_w)] = 1.
                                max_iou = 0.
                                for y_dim in val_crop_y:
                                    for x_dim in val_crop_x:
                                        patch_bbox = (int(x_dim[0]*img_w),int(y_dim[0]*img_h),int(x_dim[1]*img_w),int(y_dim[1]*img_h))
                                        iou = spill_mask[patch_bbox[1]:patch_bbox[3],patch_bbox[0]:patch_bbox[2]].mean()
                                        if iou > max_iou:
                                            max_iou = iou
                                            spill_patch = img.crop(patch_bbox)
                                
                                all_patches.append(self.preprocess(spill_patch))
                            else:
                                for y_dim in val_crop_y:
                                    for x_dim in val_crop_x:
                                        patch_bbox = (int(x_dim[0]*img_w),int(y_dim[0]*img_h),int(x_dim[1]*img_w),int(y_dim[1]*img_h))
                                        all_patches.append(self.preprocess(img.crop(patch_bbox)))

            imgs.append(torch.stack(all_patches))

        return imgs

    #def on_epoch_end(self):
    #    self.all_images, self.img_cats = shuffle(self.all_images, self.img_cats)
    
    def __getitem__(self, index):
        if self.train:
            floors_sample = random.sample(self.all_floors,3)
            self.floors = [Image.open(floor) for floor in floors_sample]

            cats = []
            frames = []
            bboxes = []
            all_bboxes = []
            vids = []
            vids_sample = np.random.choice(self.videos, 2, replace=False, p=self.yt_probs)
            for vid in vids_sample:
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

                if len(no_spill_frames) < self.batch_nums[0]//6:
                    neg_fr_idxs = []
                    neg_frames =  glob.glob('/home/petrus/SpillDetection/images/train/spill_vids/neg_frames/*.png')
                else:
                    neg_fr_idxs = random.sample(no_spill_frames,self.batch_nums[0]//6)

                pos_fr_idxs = random.sample(list(spill_frames.keys()),self.batch_nums[0]//6)
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
                    elif count in neg_fr_idxs or len(neg_fr_idxs)<self.batch_nums[0]//6:
                        if len(neg_fr_idxs)<self.batch_nums[0]//6:
                            frame_name = random.choice(neg_frames)
                        
                        if len(pos_bboxes)==0 or len(neg_fr_idxs)<self.batch_nums[0]//6:
                            frame = Image.open(frame_name)
                            lux = np.random.randint(frame.size[0]-100)
                            luy = np.random.randint(frame.size[1]-100)
                            s = np.random.randint(100)
                            bbox = (lux,luy,lux+s,luy+s)
                            neg_fr_idxs.append(frame_name)
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

            cats = []
            frames = []
            bboxes = []
            vids = np.random.choice(self.morningside, 2, replace=False, p=self.vid_probs)
            for vid in vids:
                with open(vid[:-4]+'.txt','r') as fp:
                    lines = fp.read().splitlines()

                spill_frames = defaultdict(list)
                for line in lines:
                    if ' spill' in line:
                        fr,lab,lux,luy,rbx,rby = line.split(' ')
                        spill_frames[int(fr[2:])].append((int(lux),int(luy),int(rbx),int(rby)))

                pos_fr_idxs = random.sample(list(spill_frames.keys()),self.batch_nums[0]//3)
                for fr in pos_fr_idxs:
                    frame = Image.open(vid[:-4]+'/{}.png'.format(fr))
                    frames.append(frame)
                    cats.append(0)
                    bbox = random.choice(spill_frames[fr])
                    if (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) < 100:
                        print(vid,fr,bbox)
                    bboxes.append((max(0,bbox[0]),max(0,bbox[1]),min(frame.size[0]-1,bbox[2]),min(frame.size[1]-1,bbox[3])))

                neg_vid = cv2.VideoCapture(random.choice(self.morningside_no_spill[vid]))
                for n in range(self.batch_nums[0]//3):
                    neg_vid.set(cv2.CAP_PROP_POS_FRAMES,random.randint(1,int(neg_vid.get(cv2.CAP_PROP_FRAME_COUNT))-1))
                    _,frame = neg_vid.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frames.append(frame)
                    cats.append(1)
                    bboxes.append(bboxes[-1])

            imgs = self.__get_image__(index, dataset='morningside', sampled_frames=(frames,bboxes,cats))
            pos_images.extend([img for img,cat in zip(imgs,cats) if cat==0])
            neg_images.extend([img for img,cat in zip(imgs,cats) if cat==1])
        
            for ix in range(self.batch_nums[1]):
                imgs = self.__get_image__(index+ix, dataset='spill')
                pos_images.append(imgs[0])
                neg_images.append(imgs[1])

            for ix in range(self.batch_nums[2]):
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

            for t in ['clear','dark','opaque']:
                for v,_ in self.val_frames[t].items():
                    for sample in random.sample(self.val_no_spills[t][v],3):
                        frames.append(Image.open(sample))

            '''no_spill_frames = random.sample(self.val_no_spills,FLAGS.val_batch)
            for ns_frame in no_spill_frames:
                frames.append(Image.open(ns_frame))'''

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

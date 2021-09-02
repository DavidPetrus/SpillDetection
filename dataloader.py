import numpy as np
import random
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import glob
from collections import defaultdict

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, directory, batch_size,input_size=(224, 224, 3),train=True):
        
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
        self.input_size = 720

        self.zeros = tf.zeros(128)
        self.ones = tf.ones(128)

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
            frames,bboxes,cats = sampled_frames
            all_images = frames
        elif dataset == 'spill':
            pos_img = cv2.imread(self.spill_images[np.random.randint(len(self.spill_images))])
            neg_img = cv2.imread(self.not_spill_images[np.random.randint(len(self.not_spill_images))])
            neg_img2 = cv2.imread(self.not_spill_images[np.random.randint(len(self.not_spill_images))])
            all_images = [pos_img,neg_img,neg_img2]
        elif dataset == 'puddle':
            pos_img = cv2.imread(self.puddle_images[np.random.randint(len(self.puddle_images))])
            neg_img = cv2.imread(self.not_puddle_images[np.random.randint(len(self.not_puddle_images))])
            neg_img2 = cv2.imread(self.not_puddle_images[np.random.randint(len(self.not_puddle_images))])
            all_images = [pos_img,neg_img,neg_img2]

        imgs = []
        for i_ix,img in enumerate(all_images):
            cat = 0 if i_ix==0 else 1

            img = img/255.
            img_h,img_w,_ = img.shape
            img_size = min(img_h,img_w)
            if self.train:
                #img = tf.keras.preprocessing.image.random_channel_shift(img, intensity_range=0.3, channel_axis=2)
                if dataset == 'spill':
                    if cat == 0:
                        crop_size = np.random.randint(int(0.8*img_size),img_size+1)
                        new_size = np.random.randint(int(self.input_size*0.1),int(self.input_size*0.4))
                    else:
                        if np.random.random() < 0.25:
                            crop_size = np.random.randint(int(0.1*img_size),int(0.4*img_size))
                            new_size = np.random.randint(int(self.input_size*0.1),int(self.input_size*0.4))
                        else:
                            crop_size = np.random.randint(int(0.7*img_size),img_size)
                            new_size = np.random.randint(int(self.input_size*0.8),int(self.input_size*1.4))
                elif dataset == 'video':
                    cat = cats[i_ix]
                    bbox = bboxes[i_ix]
                    lux,luy,rbx,rby = bbox
                    if rby>=img_h or rbx>=img_w or lux<=0 or luy<=0:
                        print(bbox,img.shape)
                    cr_l = np.random.randint(0,lux)
                    cr_t = np.random.randint(0,luy)
                    cr_r = np.random.randint(rbx,img_w)
                    cr_b = np.random.randint(rby,img_h)
                    crop_dim = np.array([cr_l,cr_t,cr_r,cr_b])
                    crop_size = min(crop_dim[2]-crop_dim[0],crop_dim[3]-crop_dim[1])
                    max_diff = np.argmax([lux-cr_l,luy-cr_t,cr_r-rbx,cr_b-rby])
                    if max_diff >= 2:
                        crop_dim[max_diff] = crop_dim[max_diff-2]+crop_size
                    else:
                        crop_dim[max_diff] = crop_dim[max_diff+2]-crop_size

                    crop = img[crop_dim[1]:crop_dim[3],crop_dim[0]:crop_dim[2]]
                    new_size = np.random.randint(int(img_size*self.vid_resize_range[img_size][0]),int(img_size*self.vid_resize_range[img_size][1]))
                elif dataset == 'puddle':
                    if cat == 0:
                        crop_size = np.random.randint(int(0.8*img_size),img_size+1)
                        new_size = np.random.randint(int(self.input_size*0.1),int(self.input_size*0.4))
                    else:
                        if img_size==400:
                            crop_size = np.random.randint(int(0.5*img_size),int(img_size))
                            new_size = np.random.randint(int(self.input_size*0.1),int(self.input_size*0.4))
                        else:
                            crop_size = np.random.randint(int(0.7*img_size),img_size)
                            new_size = np.random.randint(int(self.input_size*0.6),int(self.input_size*1.2))

                if dataset != 'video':
                    crop = tf.image.random_crop(img,(crop_size,crop_size,3))
                #crop = tfa.image.rotate(crop,np.random.randint(-20,20))
                img = tf.image.resize(crop, [new_size,new_size])

                img = tf.image.random_brightness(img, 0.7)
                img = tf.image.random_contrast(img, 0.3,1.7)
                img = tf.image.random_hue(img, 0.25)
                img = tf.image.random_flip_left_right(img)
                img = tf.image.resize_with_crop_or_pad(img,self.input_size,self.input_size)
            else:
                img = img[img_h//2-img_size//2:img_h//2+img_size//2,img_w//2-img_size//2:img_w//2+img_size//2]
                if dataset == 'spill':
                    if cat == 0:
                        new_size = int(self.input_size*0.2)
                        img = tf.image.resize(img, [new_size,new_size])
                    else:
                        new_size = self.input_size

                img = tf.image.resize_with_crop_or_pad(img,self.input_size,self.input_size)
                img = tf.cast(img,dtype=tf.float32)

            imgs.append(img)

        return imgs

    #def on_epoch_end(self):
    #    self.all_images, self.img_cats = shuffle(self.all_images, self.img_cats)
    
    def __getitem__(self, index):
        '''pos_puddle = []
        neg_puddle = []
        pos_video = []
        neg_video = []
        pos_spill = []
        neg_spill = []'''

        vid = self.videos[np.random.randint(len(self.videos))]
        with open(vid[:-4]+'.txt','r') as fp:
            lines = fp.read().splitlines()

        no_spill_frames = []
        spill_frames = defaultdict(list)
        spill_bboxes = []
        for line in lines:
            if 'no_spill' in line:
                fr,lab = line.split(' ')
                no_spill_frames.append(int(fr[2:]))
            elif 'spill' in line:
                fr,lab,lux,luy,rbx,rby = line.split(' ')
                #spill_frames.append((int(fr[2:]),int(lux),int(luy),int(rbx),int(rby)))
                spill_frames[int(fr[2:])].append((int(lux),int(luy),int(rbx),int(rby)))

        if len(spill_frames) >= 4:
            neg_fr_idxs = random.sample(no_spill_frames,min(len(no_spill_frames),4))
        else:
            neg_fr_idxs = random.sample(no_spill_frames,8)

        pos_fr_idxs = random.sample(list(spill_frames.keys()),8-len(neg_fr_idxs))
        pos_bboxes = []
        for fr in pos_fr_idxs:
            pos_bboxes.extend(spill_frames[fr])

        vid_frames = glob.glob(vid[:-4]+'/*.png')
        cats = []
        frames = []
        bboxes = []
        for frame_name in vid_frames:
            count = int(frame_name.split('/')[-1][:-4])
            if count in pos_fr_idxs:
                bbox = random.choice(spill_frames[count])
                cats.append(0)
            elif count in neg_fr_idxs:
                if len(pos_bboxes)==0:
                    frame = cv2.imread(frame_name)
                    lux = np.random.randint(frame.shape[1]-100)
                    luy = np.random.randint(frame.shape[0]-100)
                    s = np.random.randint(100)
                    bbox = (lux,luy,lux+s,luy+s)
                else:
                    bbox = random.choice(pos_bboxes)
                cats.append(1)
            else:
                continue

            frame = cv2.imread(frame_name)
            frames.append(frame)
            bboxes.append((max(0,bbox[0]),max(0,bbox[1]),min(frame.shape[1]-1,bbox[2]),min(frame.shape[0]-1,bbox[3])))

        imgs = self.__get_image__(index, dataset='video', sampled_frames=(frames,bboxes,cats))
        pos_images = [img for img,cat in zip(imgs,cats) if cat==0]
        neg_images = [img for img,cat in zip(imgs,cats) if cat==1]

        for ix in range(self.batch_size):
            '''if self.train:
                imgs = self.__get_image__(index+ix, dataset='puddle')
                pos_puddle.append(imgs[0])
                neg_puddle.append(imgs[1])
                neg_puddle.append(imgs[2])'''

            #if ix < 8:
            imgs = self.__get_image__(index+ix, dataset='spill')
            pos_images.append(imgs[0])
            neg_images.append(imgs[1])
            neg_images.append(imgs[2])

        '''if self.train:
            X = tf.concat([tf.stack(pos_puddle),tf.stack(pos_video),tf.stack(pos_spill),tf.stack(neg_puddle),tf.stack(neg_video),tf.stack(neg_spill)],axis=0)
        else:
            X = tf.concat([tf.stack(pos_video),tf.stack(pos_spill),tf.stack(neg_video),tf.stack(neg_spill)],axis=0)'''

        X = tf.stack(pos_images+neg_images)
        lab = tf.reshape(tf.concat([self.ones[:len(pos_images)],self.zeros[:len(neg_images)]],axis=0),[-1,1])

        return X,lab
    
    def __len__(self):
        return self.n // self.batch_size

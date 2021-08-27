import numpy as np
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import glob

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, directory, batch_size,input_size=(224, 224, 3),train=True):
        
        self.spill_images = glob.glob(directory+'/spills/*')
        self.not_spill_images = glob.glob(directory+'/not_spills/*')
        self.all_images = self.spill_images + self.not_spill_images
        self.img_cats = [0] * len(self.spill_images) + [1] * len(self.not_spill_images)
        #self.all_images, self.img_cats = shuffle(self.all_images, self.img_cats)

        self.puddle_images = glob.glob(directory+'/puddle/*')
        self.not_puddle_images = glob.glob(directory+'/not_puddle/*')
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
        self.input_size = input_size

        self.zeros = tf.zeros(128)

        self.video_groups = ['gallon','smacking_drinks','react','store1_1','store1_2','store1_3']
        self.vid_probs = [0.6,0.7,0.9,0.93,0.96,1.]
        self.vid_pos = {}
        self.vid_neg = {}
        for gr in self.video_groups:
            self.vid_pos[gr] = glob.glob(directory+'/spill_vids/{}*'.format(gr))
            self.vid_neg[gr] = glob.glob(directory+'/not_spill_vids/{}*'.format(gr))
        
        self.train = train
        if self.train:
            self.n = len(self.all_puddles)
        else:
            self.batch_size = 8
            self.n = len(self.all_images)
        

    def __get_image__(self,index,dataset=''):
        if dataset in self.video_groups:
            vid = self.vid_pos[dataset][np.random.randint(len(self.vid_pos[dataset]))]
            pos_imgs = glob.glob(vid+'/*')
            neg_imgs = glob.glob(vid.replace('spill_vids','not_spill_vids')+'_not_spill/*')

            pos_img = cv2.imread(pos_imgs[np.random.randint(len(pos_imgs))])
            neg_img = cv2.imread(neg_imgs[np.random.randint(len(neg_imgs))])
            neg_img2 = cv2.imread(neg_imgs[np.random.randint(len(neg_imgs))])
        elif dataset == 'spill':
            pos_img = cv2.imread(self.spill_images[np.random.randint(len(self.spill_images))])
            neg_img = cv2.imread(self.not_spill_images[np.random.randint(len(self.not_spill_images))])
            neg_img2 = cv2.imread(self.not_spill_images[np.random.randint(len(self.not_spill_images))])
        elif dataset == 'puddle':
            pos_img = cv2.imread(self.puddle_images[np.random.randint(len(self.puddle_images))])
            neg_img = cv2.imread(self.not_puddle_images[np.random.randint(len(self.not_puddle_images))])
            neg_img2 = cv2.imread(self.not_puddle_images[np.random.randint(len(self.not_puddle_images))])

        imgs = []
        for i_ix,img in enumerate([pos_img,neg_img,neg_img2]):
            if i_ix==0:
                cat = 0
            else:
                cat = 1

            img = img/255.
            h,w,_ = img.shape
            img_size = min(h,w)
            if self.train:
                #img = tf.keras.preprocessing.image.random_channel_shift(img, intensity_range=0.3, channel_axis=2)
                if dataset == 'spill':
                    if cat == 0:
                        crop_size = np.random.randint(int(0.7*img_size),img_size+1)
                    else:
                        crop_size = np.random.randint(int(0.1*img_size),int(0.6*img_size))
                elif dataset in self.video_groups:
                    if max(h,w) > 2*img_size:
                        crop_size = np.random.randint(int(0.9*img_size),img_size+1)   
                    else:
                        crop_size = np.random.randint(int(0.7*img_size),img_size+1)
                elif dataset == 'puddle':
                    crop_size = np.random.randint(int(0.8*img_size),img_size+1)

                crop = tf.image.random_crop(img,(crop_size,crop_size,3))
                crop = tfa.image.rotate(crop,np.random.randint(-20,20))
                if crop_size > 160:
                    new_size = np.random.randint(80,224)
                else:
                    new_size = min(max(int(crop_size*(0.7+np.random.random())),80),224)

                img = tf.image.resize(crop, [new_size,new_size])

                img = tf.image.random_brightness(img, 0.7)
                img = tf.image.random_contrast(img, 0.3,1.7)
                img = tf.image.random_hue(img, 0.25)
                img = tf.image.random_flip_left_right(img)
                img = tf.image.resize_with_crop_or_pad(img,224,224)
            else:
                if dataset == 'spill':
                    if cat == 0:
                        crop_size = np.random.randint(int(0.7*img_size),img_size+1)
                    else:
                        crop_size = np.random.randint(int(0.1*img_size),int(0.6*img_size))
                elif dataset in self.video_groups:
                    if max(h,w) > 2*img_size:
                        crop_size = np.random.randint(int(0.9*img_size),img_size+1)   
                    else:
                        crop_size = np.random.randint(int(0.7*img_size),img_size+1)
                elif dataset == 'puddle':
                    crop_size = np.random.randint(int(0.8*img_size),img_size+1)

                crop = tf.image.random_crop(img,(crop_size,crop_size,3))
                if crop_size > 160:
                    new_size = np.random.randint(80,224)
                else:
                    new_size = min(max(int(crop_size*(0.7+np.random.random())),80),224)

                img = tf.image.resize(crop, [new_size,new_size])
                img = tf.image.resize_with_crop_or_pad(img,224,224)

            imgs.append(img)

        return imgs

    def on_epoch_end(self):
        self.all_images, self.img_cats = shuffle(self.all_images, self.img_cats)
    
    def __getitem__(self, index):
        pos_puddle = []
        neg_puddle = []
        pos_video = []
        neg_video = []
        pos_spill = []
        neg_spill = []
        rand = np.random.random()
        if self.train:
            for vid_prob,vid_gr in zip(self.vid_probs,self.video_groups):
                if rand < vid_prob:
                    vid_group = vid_gr
                    break
        else:
            for vid_prob,vid_gr in zip(self.vid_probs[:2]+[1.],self.video_groups[:3]):
                if rand < vid_prob:
                    vid_group = vid_gr
                    break

        for ix in range(self.batch_size):
            if self.train:
                imgs = self.__get_image__(index+ix, dataset='puddle')
                pos_puddle.append(imgs[0])
                neg_puddle.append(imgs[1])
                neg_puddle.append(imgs[2])

            if ix < 8:
                imgs = self.__get_image__(index+ix, dataset='spill')
                pos_spill.append(imgs[0])
                neg_spill.append(imgs[1])
                neg_spill.append(imgs[2])

            if ix < 4:
                imgs = self.__get_image__(index+ix, dataset=vid_group)
                pos_video.append(imgs[0])
                neg_video.append(imgs[1])
                neg_video.append(imgs[2])

        X = tf.concat([tf.stack(pos_puddle),tf.stack(pos_video),tf.stack(pos_spill),tf.stack(neg_puddle),tf.stack(neg_video),tf.stack(neg_spill)],axis=0)

        return X,self.zeros[:X.shape[0]]
    
    def __len__(self):
        return self.n // self.batch_size

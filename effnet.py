from tensorflow.keras.applications import * #Efficient Net included here
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import os
import glob
import shutil
import random
import cv2
import datetime
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm
from tensorflow.keras import optimizers
import tensorflow as tf
import tensorflow_addons as tfa

#Use this to check if the GPU is configured correctly
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_integer('batch_size',8,'')
flags.DEFINE_integer('epochs',1000,'')
flags.DEFINE_float('lr',3*10**-5,'')
flags.DEFINE_float('dropout',0.2,'')
flags.DEFINE_float('label_smoothing',0.05,'')


class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, directory, batch_size,input_size=(224, 224, 3),train=True):
        
        self.spill_images = glob.glob(directory+'/spills/*')
        self.not_spill_images = glob.glob(directory+'/not_spills/*')
        self.all_images = self.spill_images + self.not_spill_images
        self.img_cats = [0] * len(self.spill_images) + [1] * len(self.not_spill_images)
        self.all_images, self.img_cats = shuffle(self.all_images, self.img_cats)

        self.vids = glob.glob(directory+'/spill_vids/*')
        self.vid_frames = {}
        self.vid_cats = {}
        for vid in self.vids:
            spill_frames = glob.glob(vid+'/*')
            not_spill_frames = glob.glob(vid.replace('spill_vids','not_spill_vids')+'/*')
            self.vid_frames[vid] = spill_frames + not_spill_frames
            self.vid_cats[vid] = [0] * len(spill_frames) + [1] * len(not_spill_frames)

        self.batch_size = batch_size
        self.input_size = input_size
        
        self.n = len(self.all_images)
        self.train = train

    def __get_image__(self,index,min_crop_size,val_min_crop_size,vid_image=False):
        if vid_image:
            vid = self.vids[np.random.randint(len(self.vids))]
            f_ix = np.random.randint(len(self.vid_frames[vid]))
            img = cv2.imread(self.vid_frames[vid][f_ix])
            cat = self.vid_cats[vid][f_ix]
        else:
            img = cv2.imread(self.all_images[index])
            cat = self.img_cats[index]

        if img is None:
            print(self.all_images[index])
        img = img/255.
        h,w,_ = img.shape
        if self.train:
            img = tf.keras.preprocessing.image.random_channel_shift(img, intensity_range=0.3, channel_axis=2)
            min_crop_size = min_crop_size[0] if cat==1 else min_crop_size[1]
            cr_h,cr_w = np.random.randint(int(min_crop_size*h),h+1), np.random.randint(int(min_crop_size*w),w+1)
            crop = tf.image.random_crop(img,(cr_h,cr_w,3))
            crop = tf.image.central_crop(crop,central_fraction=min(cr_h/cr_w,cr_w/cr_h))
            crop = tfa.image.rotate(crop,np.random.randint(-30,30))
            resize_frac = 0.5+np.random.random()
            #if int(cr_h*resize_frac) > self.input_size[0] or int(cr_w*resize_frac) > self.input_size[1]:
            if True:
                img = tf.image.resize_with_pad(crop, target_height=self.input_size[0], target_width=self.input_size[1])
            else:
                img = tf.image.resize_with_pad(crop, target_height=int(cr_h*resize_frac), target_width=int(cr_w*resize_frac))
                img = tf.image.pad_to_bounding_box(img, min((self.input_size[0]-img.shape[0])//2,20), min((self.input_size[1]-img.shape[1])//2,20), \
                    self.input_size[0], self.input_size[1])

            img = tf.image.random_brightness(img, 0.7)
            img = tf.image.random_contrast(img, 0.3,1.7)
            img = tf.image.random_hue(img, 0.25)
            img = tf.image.random_flip_left_right(img)
        else:
            min_crop_size = val_min_crop_size[0] if cat==1 else val_min_crop_size[1]
            cr_h,cr_w = np.random.randint(int(min_crop_size*h),h+1), np.random.randint(int(min_crop_size*w),w+1)
            crop = tf.image.random_crop(img,(cr_h,cr_w,3))
            crop = tf.image.central_crop(crop,central_fraction=min(cr_h/cr_w,cr_w/cr_h))
            #if cr_h > self.input_size[0] or cr_w > self.input_size[1]:
            if True:
                img = tf.image.resize_with_pad(crop, target_height=self.input_size[0], target_width=self.input_size[1])
            else:
                img = tf.image.resize_with_pad(crop, target_height=cr_h, target_width=cr_w)
                img = tf.image.pad_to_bounding_box(img, min((self.input_size[0]-img.shape[0])//2,20), min((self.input_size[1]-img.shape[1])//2,20), \
                    self.input_size[0], self.input_size[1])
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8,1.2)

        lab = np.array([1-cat])
        return img,lab

    def on_epoch_end(self):
        self.all_images, self.img_cats = shuffle(self.all_images, self.img_cats)
    
    def __getitem__(self, index):
        images = []
        labels = []
        for ix in range(self.batch_size):
            img,target = self.__get_image__(index+ix,min_crop_size=(0.2,0.6),val_min_crop_size=(0.2,1.))
            images.append(img)
            labels.append(target)

            img,target = self.__get_image__(index+ix,min_crop_size=(0.75,0.75),val_min_crop_size=(0.75,0.75),vid_image=True)
            images.append(img)
            labels.append(target)

        images = np.stack(images)
        labels = np.stack(labels)

        return images,labels
    
    def __len__(self):
        return self.n // self.batch_size


def main(argv):

    TRAIN_IMAGES_PATH = "./images/train"
    VAL_IMAGES_PATH = "./images/val"
    NUMBER_OF_TRAINING_IMAGES = len(glob.glob(TRAIN_IMAGES_PATH+"/*/*"))-2
    NUMBER_OF_VALIDATION_IMAGES = len(glob.glob(VAL_IMAGES_PATH+"/*/*"))-2
    print("Num train:",NUMBER_OF_TRAINING_IMAGES)
    print("Num val:",NUMBER_OF_VALIDATION_IMAGES)
    batch_size = FLAGS.batch_size
    height = 224
    width = 224
    epochs = FLAGS.epochs

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for dev in physical_devices:
        config = tf.config.experimental.set_memory_growth(dev, True)

    conv_base = EfficientNetB0(weights="effnet_weights/efficientnetb0_notop.h5", include_top=False, input_shape=(height,width,3))

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalMaxPooling2D(name="gap"))
    #avoid overfitting
    model.add(layers.Dropout(rate=FLAGS.dropout, name="dropout_out"))
    # Set NUMBER_OF_CLASSES to the number of your final predictions.
    model.add(layers.Dense(1, activation="sigmoid", name="fc_out"))
    #conv_base.trainable = False

    train_generator = CustomDataGen(TRAIN_IMAGES_PATH, batch_size, train=True)
    validation_generator = CustomDataGen(VAL_IMAGES_PATH, batch_size, train=False)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=optimizers.Adam(lr=FLAGS.lr),
        metrics=["acc"],
    )

    checkpoint = ModelCheckpoint('effnet_weights/'+FLAGS.exp+'.h5', monitor='val_loss', verbose=1, save_best_only=False, mode='min', period=1)

    history = model.fit(
        train_generator,
        steps_per_epoch=NUMBER_OF_TRAINING_IMAGES // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=NUMBER_OF_VALIDATION_IMAGES // batch_size,
        verbose=1,
        use_multiprocessing=False,
        workers=8,
        callbacks=[checkpoint])

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)



'''history = model.fit_generator(
    train_generator,
    steps_per_epoch=NUMBER_OF_TRAINING_IMAGES // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=NUMBER_OF_VALIDATION_IMAGES // batch_size,
    verbose=1,
    use_multiprocessing=False,
    workers=16,
    callbacks=[checkpoint]
)'''



'''train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=60,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="constant",
    preprocessing_function=color_distort,
)

#test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    TRAIN_IMAGES_PATH,
    # All images will be resized to target height and width.
    target_size=(height, width),
    batch_size=batch_size,
    # Since we use categorical_crossentropy loss, we need categorical labels
    class_mode="categorical",
)
validation_generator = test_datagen.flow_from_directory(
    VAL_IMAGES_PATH,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode="categorical",
)'''
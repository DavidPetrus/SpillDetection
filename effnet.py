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
flags.DEFINE_integer('batch_size',20,'')
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

        self.puddle_images = glob.glob(directory+'/puddle/*')
        self.not_puddle_images = glob.glob(directory+'/not_puddle/*')
        self.all_puddles = self.puddle_images + self.not_puddle_images
        self.puddle_cats = [0] * len(self.puddle_images) + [1] * len(self.not_puddle_images)
        self.all_puddles, self.puddle_cats = shuffle(self.all_puddles, self.puddle_cats)

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
        
        self.train = train
        if self.train:
            self.n = len(self.all_puddles)
        else:
            self.batch_size = 9
            self.n = len(self.all_images)
        

    def __get_image__(self,index,dataset=''):
        if dataset == 'video':
            vid = self.vids[np.random.randint(len(self.vids))]
            f_ix = np.random.randint(len(self.vid_frames[vid]))
            img = cv2.imread(self.vid_frames[vid][f_ix])
            cat = self.vid_cats[vid][f_ix]
        elif dataset == 'spill':
            s_ix = np.random.randint(len(self.all_images))
            img = cv2.imread(self.all_images[s_ix])
            cat = self.img_cats[s_ix]
        elif dataset == 'puddle':
            img = cv2.imread(self.all_puddles[index])
            cat = self.puddle_cats[index]

        if img is None:
            print(self.all_images[index])
        img = img/255.
        h,w,_ = img.shape
        img_size = min(h,w)
        if self.train:
            img = tf.keras.preprocessing.image.random_channel_shift(img, intensity_range=0.3, channel_axis=2)
            if dataset == 'spill':
                if cat == 0:
                    crop_size = np.random.randint(int(0.7*img_size),img_size+1)
                else:
                    crop_size = np.random.randint(int(0.1*img_size),int(0.6*img_size))
            elif dataset == 'video':
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
            elif dataset == 'video':
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

        lab = np.array([1-cat])
        return img,lab

    def on_epoch_end(self):
        self.all_images, self.img_cats = shuffle(self.all_images, self.img_cats)
    
    def __getitem__(self, index):
        images = []
        labels = []
        for ix in range(self.batch_size):
            if self.train:
                img,target = self.__get_image__(index+ix, dataset='puddle')
                images.append(img)
                labels.append(target)

            if ix < 8:
                img,target = self.__get_image__(index+ix, dataset='spill')
                images.append(img)
                labels.append(target)

            if ix < 4:
                img,target = self.__get_image__(index+ix, dataset='video')
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
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

from dataloader import CustomDataGen
from model import CustomModel

import wandb
from wandb.keras import WandbCallback

#Use this to check if the GPU is configured correctly
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#for dev in physical_devices:
#    config = tf.config.experimental.set_memory_growth(dev, True)

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_integer('batch_size',8,'')
flags.DEFINE_integer('epochs',1000,'')
flags.DEFINE_float('lr',3*10**-4,'')
flags.DEFINE_float('cls_dropout',0.25,'')
flags.DEFINE_float('loc_dropout',0.15,'')
flags.DEFINE_float('label_smoothing',0.05,'')
flags.DEFINE_float('min_crop_size',0.1,'')
flags.DEFINE_float('max_crop_size',0.3,'')

flags.DEFINE_integer('num_prototypes',20,'')
flags.DEFINE_integer('top_k',3,'')
flags.DEFINE_float('gamma',300,'')
flags.DEFINE_float('all_margin',0.25,'')
#flags.DEFINE_float('puddle_margin',0.3,'')
#flags.DEFINE_float('vid_margin',0.2,'')
#flags.DEFINE_float('puddle_coeff',0.1,'')
#flags.DEFINE_float('vid_coeff',0.5,'')


def main(argv):

    wandb.init(project="SpillDetection",name=FLAGS.exp)
    wandb.config.update(flags.FLAGS)

    TRAIN_IMAGES_PATH = "./images/train"
    VAL_IMAGES_PATH = "./images/val"
    NUMBER_OF_TRAINING_IMAGES = len(glob.glob(TRAIN_IMAGES_PATH+"/spills/*"))
    NUMBER_OF_VALIDATION_IMAGES = len(glob.glob(VAL_IMAGES_PATH+"/spills/*"))
    print("Num train:",NUMBER_OF_TRAINING_IMAGES)
    print("Num val:",NUMBER_OF_VALIDATION_IMAGES)
    batch_size = FLAGS.batch_size
    height = 224
    width = 224
    epochs = FLAGS.epochs

    spill_classifier = CustomModel()

    train_generator = CustomDataGen(TRAIN_IMAGES_PATH, batch_size, train=True)
    validation_generator = CustomDataGen(VAL_IMAGES_PATH, batch_size, train=False)

    spill_classifier.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=optimizers.Adam(lr=FLAGS.lr)
    )

    checkpoint = ModelCheckpoint('effnet_weights/'+FLAGS.exp+'.h5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=False, mode='min', period=1)

    history = spill_classifier.fit(
        train_generator,
        steps_per_epoch=NUMBER_OF_TRAINING_IMAGES // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=NUMBER_OF_VALIDATION_IMAGES // 4,
        verbose=1,
        use_multiprocessing=False,
        workers=8,
        callbacks=[WandbCallback(save_model=False),checkpoint])

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
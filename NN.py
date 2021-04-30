#import the generator
import generator
import PIETZ_UNET

#import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import argparse
from imutils import paths

from trainingmonitor import TrainingMonitor
import cv2
import os
import math
import random


#keras import and windows InteractiveSession bug fix
from tensorflow.keras.optimizers import SGD
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics as ms
from tensorflow.keras import  Model
from tensorflow.keras.layers import Input,Conv2D, Concatenate, MaxPooling2D,Activation,concatenate,AveragePooling2D
from tensorflow.keras.layers import UpSampling2D, Dropout, BatchNormalization,Flatten,Dense



config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)











imgs_folder_name='imgs'
masks_folder_name='masks'

train,test,trainy,testy = train_test_split(list(paths.list_images(imgs_folder_name))[:8000],list(paths.list_images(masks_folder_name))[:8000],test_size=0.1)

print("Number of training images =")
print(len(train))

print("Number of testing images =")
print(len(test))




















train_gen=generator.ImageMaskAUGGenerator(train,trainy,batchSize=20,img_width=128, img_height=128, rot_angle=5, mask_ratio=1.0, zoom_ratio=5.0, horisontal_flip=True, interior_crop=False, 
				 LAB=False, GREY=False, GaussBlur=False, mean_extraction=True, equalize=True).generator()
test_gen=generator.ImageMaskAUGGenerator(test,testy,batchSize=20,img_width=128, img_height=128, rot_angle=5, mask_ratio=1.0, zoom_ratio=5.0, horisontal_flip=True, interior_crop=False, 
				 LAB=False, GREY=False, GaussBlur=False, mean_extraction=True, equalize=True).generator()

#  uncomment this part and ##cv2.imshows in generator.py to see image and mask transformation functions at work
#while(True):
	#next(train_gen)

#Initialization of NN model
model=PIETZ_UNET.UNet((128, 128,3), out_ch=5, start_ch=32, depth=4, inc_rate=2., activation='relu', dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False)








#model=load_model('themodel')
#opt = SGD(lr=0.01, momentum=0.9, nesterov=True)

model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])


from tensorflow.keras.utils import plot_model
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
import os
plot_model(model, to_file="model_architecture.png", show_shapes=True)








# construct the set of callbacks
callbacks = [TrainingMonitor('Learning_rate.png')]

print('doshlo')
model.fit(
	train_gen,
	steps_per_epoch=360,
	validation_data=test_gen,
	validation_steps=40,
	epochs=5	,
	max_queue_size=1,
	callbacks=callbacks, verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save('themodel', overwrite=True)

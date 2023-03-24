# importing python libraries
import tensorflow as tf
from keras import optimizers
from keras.utils import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications import ResNet50, DenseNet201
from keras.applications import resnet, densenet
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

# calling the path of train validation and testing data
train_path = "data/train"
valid_path = "data/valid"
test_path = "data/test"

# ResNet 50 Model
image_shape = (460,460,3)
N_CLASSES = 1
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(dtype='float32')
train_generator = train_datagen.flow_from_directory(train_path, batch_size = BATCH_SIZE, target_size = (460,460), class_mode = 'categorical')

valid_datagen = ImageDataGenerator(dtype='float32')
valid_generator = valid_datagen.flow_from_directory(valid_path, batch_size = BATCH_SIZE, target_size = (460,460), class_mode = 'categorical')

test_datagen = ImageDataGenerator(dtype='float32')
test_generator = test_datagen.flow_from_directory(test_path, batch_size = BATCH_SIZE, target_size = (460,460), class_mode = 'categorical')

res_model = ResNet50(include_top=False, pooling='avg', weights='imagenet', input_shape = (image_shape))

# make all layers except conv5 layers not trainable
for layer in res_model.layers: 
    if 'conv5' not in layer.name: layer.trainable = False
    
# creation of the model with each layers
resnet_model = Sequential()
resnet_model.add(res_model)
resnet_model.add(Dropout(0.4))
resnet_model.add(Flatten())
resnet_model.add(BatchNormalization())
resnet_model.add(Dropout(0.4))
resnet_model.add(Dense(N_CLASSES, activation='softmax'))
resnet_model.summary()

# compiling the model 
optimizer = optimizers.Adam(learning_rate= 0.00001, decay= 1e-5)
resnet_model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics = ['acc'])

checkpointer = ModelCheckpoint(filepath='./ResNet50_model.hdf5', monitor='val_loss', verbose = 1, save_best_only=True)
early_stopping = EarlyStopping(verbose=1, patience=20)

history_res = resnet_model.fit(train_generator, steps_per_epoch = 20, epochs = 32, verbose = 1, validation_data = valid_generator, callbacks = [checkpointer, early_stopping])
    
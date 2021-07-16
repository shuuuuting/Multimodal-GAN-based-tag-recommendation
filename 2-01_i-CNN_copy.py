#%%----------------------------------------------------------------------------------------
#如有照順序從第1大項重新跑的話，可以略過此區塊，不用從資料庫取資料
#------------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import csv
from pymongo import MongoClient

conn = MongoClient('localhost', 27017) #連結mongodb
db = conn.NiusNews2020_04_12 #create database
train_id = db['split_indices'].find_one({"dataname":"train"})['indexlist']
test_id = db['split_indices'].find_one({"dataname":"test"})['indexlist']

##### VGG #####
#%% preprocess
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import VGG19
from keras.models import Model
import cv2
from PIL import Image
from gridfs import *
import io
gridFS = GridFS(db, collection="fs")
#load the model
#base_model = VGG16(weights='imagenet')
vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))
# Freeze four convolution blocks
for layer in vgg_model.layers[:17]:
    layer.trainable = False
# Make sure you have frozen the correct layers
for i, layer in enumerate(vgg_model.layers):
    print(i, layer.name, layer.trainable)

image_data_train = []
count = 0
print('Processing train image:')
#load image from db
for idx in train_id:
    if count%20==0:
        print(count)
    count+=1
    image_data = gridFS.find_one({"filename": str(idx)})
    #convert the image pixels to a numpy array
    image = np.array(Image.open(io.BytesIO(image_data.read())).convert('RGB'))
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    #img = Image.fromarray(image)
    #img.show()
    #reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #image = image/255.0
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    image_data_train.append(image)
image_data_train = np.vstack(image_data_train)

image_data_test = []
count = 0
print('Processing test image:')
#load image from db
for idx in test_id:
    if count%20==0:
        print(count)
    count+=1
    image_data = gridFS.find_one({"filename": str(idx)})
    #convert the image pixels to a numpy array
    image = np.array(Image.open(io.BytesIO(image_data.read())).convert('RGB'))
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    #image = image/255.0
    #img = Image.fromarray(image)
    #img.show()
    #reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    image_data_test.append(image)
image_data_test = np.vstack(image_data_test)

#%% model
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Activation, Flatten, Reshape, RepeatVector
from keras.layers.convolutional import AveragePooling1D,Conv1D
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input, Dense, Embedding, Dropout, Lambda, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras_self_attention import SeqSelfAttention

num_region = 7*7
embedding_size = 200 
seq_length = 50
TopK = 10
batch_size = 64
num_tags = tag_train.shape[1]
optimizer = Adam(lr=0.0005, decay=0.000001)

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True) #持續5個epoch沒下降就停

def imagecnn():
    x = vgg_model.output
    x = Reshape(target_shape=(num_region, 512))(x)
    x = Dense(embedding_size, activation="tanh", use_bias=False, name='icnn')(x) 
    #x = SeqSelfAttention(attention_activation='sigmoid')(x)
    #x.set_shape((x.shape[0],num_region,embedding_size))
    x = AveragePooling1D(pool_size=num_region//2, padding="same")(x)
    x = Flatten()(x)
    #model.add(GlobalMaxPooling1D())
    x = Dense(600, activation='relu')(x)
    x = Dense(num_tags, activation='sigmoid')(x)
    model = Model(inputs=vgg_model.input, outputs=x)
    model.compile(loss='BinaryCrossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

model = imagecnn()
model.fit(image_data_train, tag_train.astype(np.float32), epochs=50, batch_size=batch_size, validation_data=(image_data_test, tag_test), callbacks=[early_stopping])
#model.save('pretrained_iCNN.h5')
y_pred = model.predict(image_data_test)
acc_K, precision_K, recall_K, f1_K = evaluation(tag_test, y_pred, TopK)
print('acc: ', acc_K)
print('precision: ', precision_K)
print('recall: ', recall_K)
print('f1: ', f1_K)

#451個tag
#acc:  0.5161036920659858
#precision:  0.09442262372348782
#recall:  0.3519853738824673
#f1:  0.14126520669078776

#263個tag
#acc:  0.58994708994709
#precision:  0.10696649029982365
#recall:  0.447263864393494
#f1:  0.16350414341593272

#flatten -> GlobalAveragePooling
#add self-attention after conv1d
#acc:  0.5970017636684304
#precision:  0.10652557319223986
#recall:  0.4635137594132303
#f1:  0.16392006246708363

#flatten -> GlobalMaxPooling
#add self-attention after conv1d
#acc:  0.6111111111111112
#precision:  0.10943562610229278
#recall:  0.47721018448531677
#f1:  0.16852849455659585

##### ResNet #####
#%% preprocess
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
image_train = []
count = 0
for idx in train_id:
    if count%20==0:
        print(count)
    count+=1
    image_data = gridFS.find_one({"filename": str(idx)})
    #convert the image pixels to a numpy array
    image = np.array(Image.open(io.BytesIO(image_data.read())).convert('RGB'))
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    image = image/255.0
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image_train.append(image)
image_train = np.vstack(image_train)

image_test = []
count = 0
print('Processing test image:')
#load image from db
for idx in test_id:
    if count%20==0:
        print(count)
    count+=1
    image_data = gridFS.find_one({"filename": str(idx)})
    #convert the image pixels to a numpy array
    image = np.array(Image.open(io.BytesIO(image_data.read())).convert('RGB'))
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    image = image/255.0
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image_test.append(image)
image_test = np.vstack(image_test)

#%% model
model = Sequential()
model.add(ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(image.shape[1],image.shape[2],3)))
model.add(Flatten())
model.add(Dense(600, activation='relu'))
model.add(Dense(num_tags, activation='sigmoid', name='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(image_train, tag_train, epochs=50, batch_size=batch_size)
y_pred = model.predict(image_test)
acc_K, precision_K, recall_K, f1_K = evaluation(tag_test, y_pred, TopK)
print('acc: ', acc_K)
print('precision: ', precision_K)
print('recall: ', recall_K)
print('f1: ', f1_K)

#20 epochs
#acc:  0.2812254516889238
#precision:  0.05412411626080126
#recall:  0.1815630494145812
#f1:  0.07959041938710161

#50 epochs
#acc:  0.46472663139329806
#precision:  0.08253968253968254
#recall:  0.3309208868732678
#f1:  0.1248698842411473
# %%

#%%----------------------------------------------------------------------------------------
#取出text&tag資料
#------------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import csv
from pymongo import MongoClient

conn = MongoClient('localhost', 27017) #連結mongodb
db = conn.NiusNews2020_04_12 #create database
postsdb = db['post_Jieba_ID50_tagged']
posts = pd.DataFrame(list(postsdb.find({},{"_id": 0,"postid": 1,"content_jieba": 1,"chanel_num":1})))
tags = pd.DataFrame(list(postsdb.find({},{"_id": 0,"postid": 1,"tag": 1})))

#%%----------------------------------------------------------------------------------------
#把t發生次數多於15的tag存進label.txt中(只用train data的tag計算)
#------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
posts['chanel_num'] = posts['chanel_num'].replace('0','18')
text_train, text_test, tag_train, tag_test = train_test_split(posts, tags, test_size=0.2, random_state=20, stratify=posts['chanel_num'])
#random state origin:11
text_train = text_train.reset_index(drop=True)
text_test = text_test.reset_index(drop=True)
tag_train = tag_train.reset_index(drop=True)
tag_test = tag_test.reset_index(drop=True)
toptags = pd.concat([pd.Series(x) for x in tag_train.tag], axis=0).value_counts()
toptags = toptags[toptags>15]
print('# of tag appears above 15: ', len(toptags)) #超過10次:451個 / 超過15次:263個 #stratify:超過10次:484個 / 超過15次:265個
datadir = 'C:/Users/shuting/Desktop/論文/data/'
filename = 'labels.txt'
if filename not in os.listdir(datadir):
    f = open(datadir+filename, 'w', encoding="utf-8")
with open(datadir+filename, 'w', newline='', encoding="utf-8") as txtfile:
    for row in toptags.index:
        txtfile.write(row+'\n')
txtfile.close()
labelfile = db['labels']
labelfile.drop() #確定資料庫為空的
f = open(datadir+filename, encoding='utf8')
i = 0
for line in f.readlines():
    labelfile.insert_one({'id':i,'labels':line})
    i += 1

#%%----------------------------------------------------------------------------------------
#1)把tag轉成one-hot形式
#------------------------------------------------------------------------------------------
#backup tag data
tag_tr_pd = tag_train
tag_te_pd = tag_test
tag_tr_pd = tag_tr_pd.reset_index(drop=True)
tag_te_pd = tag_te_pd.reset_index(drop=True)

tag_train['label'] = [list() for x in range(len(tag_train.index))]
tag_test['label'] = [list() for x in range(len(tag_test.index))]
tag_train[toptags.index] = pd.DataFrame([[0]*len(toptags)], index=tag_train.index)
tag_test[toptags.index] = pd.DataFrame([[0]*len(toptags)], index=tag_test.index)
for i in range(len(tag_train)):
    for t in tag_train.tag[i]:
        if t in tag_train.columns:
            tag_train[t][i] = 1
            tag_train['label'][i].append(t)
for i in range(len(tag_test)):
    for t in tag_test.tag[i]:
        if t in tag_test.columns:
            tag_test[t][i] = 1
            tag_test['label'][i].append(t)

#%%----------------------------------------------------------------------------------------
#2)把過濾完沒有tag的row刪掉
#------------------------------------------------------------------------------------------
num_tag_train = [len(x) for x in tag_train['label']]
zero_tag_train = [i for i, e in enumerate(num_tag_train) if e == 0]
text_train.drop(zero_tag_train, inplace=True)
tag_train.drop(zero_tag_train, inplace=True)
num_tag_test = [len(x) for x in tag_test['label']]
zero_tag_test = [i for i, e in enumerate(num_tag_test) if e == 0]
text_test.drop(zero_tag_test, inplace=True)
tag_test.drop(zero_tag_test, inplace=True)
#transform tag pd to np array 
tag_train = np.array(tag_train.drop(['postid','tag','label'],axis=1))
tag_test = np.array(tag_test.drop(['postid','tag','label'],axis=1))

train_id = list(text_train.postid)
test_id = list(text_test.postid)
indices = db['split_indices'] 
indices.drop()
indices.insert_many([{"dataname":'train',"indexlist":train_id},
{"dataname":'test',"indexlist":test_id}])

tag_tr_pd.index = tag_tr_pd.postid
tag_te_pd.index = tag_te_pd.postid
tag_tr_pd = tag_tr_pd.loc[train_id]
tag_te_pd = tag_te_pd.loc[test_id]

#%%----------------------------------------------------------------------------------------
#3)save text,tag spilt data to database
#------------------------------------------------------------------------------------------
text_train_db = db['text_train_db'] #create text train collection  
text_train_db.drop() #確定資料庫為空的
text_train_db.insert_many(text_train.to_dict(orient='records'))
print('共有%s筆資料' % text_train_db.count()) #5215 #4633

text_test_db = db['text_test_db'] #create text test collection  
text_test_db.drop() #確定資料庫為空的
text_test_db.insert_many(text_test.to_dict(orient='records'))
print('共有%s筆資料' % text_test_db.count()) #1273 #1134

tag_train_db = db['tag_train_db'] #create text train collection  
tag_train_db.drop() #確定資料庫為空的
tmp = pd.DataFrame(tag_train)
tmp.columns = [str(i) for i in tmp.columns]
tag_train_db.insert_many(tmp.to_dict(orient='records'))
print('共有%s筆資料' % tag_train_db.count()) #5215 #4633

tag_test_db = db['tag_test_db'] #create text test collection  
tag_test_db.drop() #確定資料庫為空的
tmp = pd.DataFrame(tag_test)
tmp.columns = [str(i) for i in tmp.columns]
tag_test_db.insert_many(tmp.to_dict(orient='records'))
print('共有%s筆資料' % tag_test_db.count()) #1273 #1134

#%%----------------------------------------------------------------------------------------
#4)save img spilt data to database
#------------------------------------------------------------------------------------------
from gridfs import *
gridFS = GridFS(db, collection="fs")
img_train_db = db['img_train_db'] #create img train collection  
img_train_db.drop() #確定資料庫為空的
for idx in train_id:
    image_data = list(gridFS.find_one({"filename": str(idx)}))
    mydict = {"filename":str(idx),"imgbytelist":image_data}
    img_train_db.insert(mydict)
print('共有%s筆資料' % img_train_db.count())

img_test_db = db['img_test_db'] #create img train collection  
img_test_db.drop() #確定資料庫為空的
for idx in test_id:
    image_data = list(gridFS.find_one({"filename": str(idx)}))
    mydict = {"filename":str(idx),"imgbytelist":image_data}
    img_test_db.insert(mydict)
print('共有%s筆資料' % img_test_db.count())

#%%----------------------------------------------------------------------------------------
#load spilt data from database
#檢查用，看放進去的資料取出有沒有問題
#------------------------------------------------------------------------------------------
text_train_db_out = pd.DataFrame(list(db['text_train_db'].find({},{"_id": 0,"postid": 1,"content_jieba": 1})))
text_test_db_out = pd.DataFrame(list(db['text_test_db'].find({},{"_id": 0,"postid": 1,"content_jieba": 1})))
tag_train_db_out = np.array(pd.DataFrame(list(db['tag_train_db'].find({},{"_id": 0}))))
tag_test_db_out = np.array(pd.DataFrame(list(db['tag_test_db'].find({},{"_id": 0}))))
img_train_db_out = pd.DataFrame(list(db['img_train_db'].find({},{"_id": 0,"filename": 1,"imgbytelist": 1})))
img_test_db_out = pd.DataFrame(list(db['img_test_db'].find({},{"_id": 0,"filename": 1,"imgbytelist": 1})))

#check if it's match
from PIL import Image
import io
train_img_0_db_out = np.array(Image.open(io.BytesIO(img_train_db_out.imgbytelist[0][0])).convert('RGB'))
train_img_0 = np.array(Image.open(io.BytesIO(gridFS.find_one({"filename": train_id[0]}).read())).convert('RGB'))

# %%

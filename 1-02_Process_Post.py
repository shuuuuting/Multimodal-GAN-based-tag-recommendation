#%%
import os
import numpy as np
import pandas as pd
import csv
from pymongo import MongoClient
import jieba
import jieba.analyse
import re
from bs4 import BeautifulSoup
import shutil

conn = MongoClient('localhost', 27017) #連結mongodb
db = conn.NiusNews202010 #create database

datadir = 'C:/Users/shuting/Desktop/論文/data/'
postid_df = pd.read_csv(datadir+'img_postid.csv', header=None)
#postdir = datadir + '/posts_202004-202010/'
postdir = datadir + '/posts_202004-12/'
#preprocess_postdir = datadir + '/posts_202010/'
preprocess_postdir = datadir + '/posts_new/'
if not os.path.exists(preprocess_postdir):
    os.makedirs(preprocess_postdir)
'''
else: 
    shutil.rmtree(preprocess_postdir)
    os.makedirs(preprocess_postdir)
'''
#%%----------------------------------------------------------------------------------------
#把網頁格式過濾掉，主要是將post裡面的資料過濾到只剩單純的新聞文字
#------------------------------------------------------------------------------------------
print('Start process posts!!')
count = 0
#將網頁格式過濾掉
for post in postid_df.values: #共4991則
    count += 1
    file = str(post[0])+'.txt'
    ori_file = open(os.path.join(postdir, file),"r",encoding="utf-8")
    text = ori_file.read()
    ori_file.close()
    soup = BeautifulSoup(str(text))
    new_file = open(os.path.join(preprocess_postdir, file),"w",encoding="utf-8")
    new_file.write(str(soup.get_text()))
    new_file.close()

#%%----------------------------------------------------------------------------------------
#將文章檔案處理成Dataframe 格式:標題 標籤 頻道編號 作者編號 上線時間 內文
#------------------------------------------------------------------------------------------
#資料儲存的List
data = []
#讀取路徑中每個檔案並做處理
for post in os.listdir(preprocess_postdir):
    content = []
    loadFile = open(os.path.join(preprocess_postdir, post),'r',encoding="utf-8")
    #Post檔內容為 標題 標籤 頻道編號 作者編號 上線時間 內文
    text = loadFile.readlines()
    for line in text[5:]: #內文
        word = re.sub("[A-Za-z0-9\[\–\-\`\～\!\@\#\＃\$\^\&\*\「\」\，\。\、\？\！\．\《\》\＼\／\（\）\(\)\=\|\{\}\'\:\;\"\：\,\[\]\.\<\>\/\?\~\！\@\#\\\&\*\%\\xa0\\n]", "", line)#過濾字元
        if word!='': 
            content.append(word)
     
    data.append({'postid':post.split('.')[0],'title':text[0].strip(),'tag':text[1].strip(),'chanel_num':text[2].strip(),'author_num':text[3].strip(),'time':text[4].strip(),'content':''.join(content)})
    loadFile.close()
    
#轉成DataFrame格式
post_df = pd.DataFrame(data, columns = ['postid','title','tag','chanel_num','author_num','time','content'])
post_df['tag'] = post_df['tag'].str.replace('#','')
post_df['tag'] = post_df['tag'].str.replace('＃','')
post_df['tag'] = post_df['tag'].str.split(',')
#del data    

#%%----------------------------------------------------------------------------------------
#Title Jieba斷詞處理
#------------------------------------------------------------------------------------------
allowPOS = ['n','v','a','d'] #篩選詞性 名詞,動詞,形容詞,副詞
jieba.analyse.set_stop_words(datadir+'stopwords.txt')
title = []
for i in range(len(post_df)):
    title.append(jieba.analyse.extract_tags(post_df['title'].loc[i], withWeight=False, allowPOS=allowPOS))
post_df['title'] = title
    
#儲存pickle備份
post_df.to_pickle(datadir+'post_preprocess.pickle')#可提供CKIP斷詞

#%%----------------------------------------------------------------------------------------
#Content Jieba斷詞處理
#------------------------------------------------------------------------------------------
#取前k個TfIdf最高的詞 調整topN來計算不同的TFIDF k
tfidf_num = 50 #ID50
print('Start Jieba ID%s' %tfidf_num)
jieba.analyse.set_stop_words('stopwords.txt') #載入停用字資料
content = []
for i in range(len(post_df)):
    if i%200==0:
        print(i)
    content.append(jieba.analyse.extract_tags(post_df['content'].loc[i], topK=tfidf_num, withWeight=False, allowPOS=allowPOS))
post_df['content_jieba'] = content

#儲存pickle備份
post_df.to_pickle(datadir+'post_Jieba_ID%s.pickle' % tfidf_num)

#%%----------------------------------------------------------------------------------------
#將處理完的文章放入db中
#------------------------------------------------------------------------------------------
conn = MongoClient('localhost', 27017) #連結mongodb
#db = conn.NiusNews202010 #create database
db = conn.NiusNews2020_04_12 #create database
post_Jieba = db['post_Jieba_ID'+str(tfidf_num)] #create collection  

post_Jieba.drop() #確定資料庫為空的
post_Jieba.insert_many(post_df.to_dict(orient='records'))
print('共有%s筆資料' % post_Jieba.count())

#%%----------------------------------------------------------------------------------------
#把有tag也有圖片的資料存到db
#------------------------------------------------------------------------------------------
#共4928則 #9357則(新)
post_df.tag = post_df.tag.map(lambda x: list(filter(None, x))) #把['']變成[]
havetag_df = post_df[post_df.tag!=post_df.tag*0]

#post textual data
post_Jieba_tagged = db['post_Jieba_ID'+str(tfidf_num)+'_tagged'] #create collection  

post_Jieba_tagged.drop() #確定資料庫為空的
post_Jieba_tagged.insert_many(havetag_df.to_dict(orient='records'))
print('共有%s筆資料' % post_Jieba_tagged.count())

#存有tag也有img資料的post
filename = 'img_tag_postid.csv'
if filename not in os.listdir(datadir):
    f = open(datadir+filename, 'w')
with open(datadir+filename, 'w', newline='') as csvfile:
    havetag_df.postid.to_csv(csvfile, index = False, header=False)
csvfile.close()

#post visual data
from gridfs import *
newimgdir = datadir +'/post_img_new/'
db['fs.chunks'].drop()
db['fs.files'].drop()
for image in list(havetag_df.postid): #遍歷圖片目錄集合
    filesname = newimgdir + image + '.jpg'
    datatmp = open(filesname, 'rb') #類似於建立檔案
    imgput = GridFS(db) #建立寫入流
    insertimg = imgput.put(datatmp, content_type='jpg', filename=image) #將資料寫入，檔案型別和名稱通過前面的分割得到
    datatmp.close()


# %%

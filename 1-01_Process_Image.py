#%%
import os, shutil
import numpy as np
import pandas as pd
import csv
from pymongo import MongoClient
from gridfs import *

datadir = 'C:/Users/shuting/Desktop/論文/data/'
#imgdir = datadir + '/images_202010/'
imgdir = datadir + '/images_202004-12/'
#newimgdir = datadir +'/post_img_202010/' 
newimgdir = datadir +'/post_img_new/' 
if not os.path.exists(newimgdir):
    os.makedirs(newimgdir)

#%%----------------------------------------------------------------------------------------
#把所有圖片檔案夾中的文章首頁圖片取出，全部放入新的資料夾，命名為postid
#------------------------------------------------------------------------------------------
#只有4991有照片
#只有9530有照片(新)
for folder in os.listdir(imgdir):
    for img in os.listdir(imgdir+folder):
        if img.find('posts_image')!=-1: #圖片名稱包含'posts_image'
            shutil.copy(imgdir+folder+'/'+img, newimgdir)
            os.rename(newimgdir+img, newimgdir+folder+'.jpg')

#%%----------------------------------------------------------------------------------------
#用一個csv檔紀錄所有圖片檔案名稱(也就是postid)
#------------------------------------------------------------------------------------------
#只收集4991/9530則post的id存入img_postid.csv 
img_postIDs = pd.DataFrame([i.split('.')[0] for i in os.listdir(newimgdir)]) #把.jpg刪掉
filename = 'img_postid.csv'
if filename not in os.listdir(datadir):
    f = open(datadir+filename, 'w')
with open(datadir+filename, 'w', newline='') as csvfile:
    img_postIDs.to_csv(csvfile, index = False, header=False)
csvfile.close()

#%%----------------------------------------------------------------------------------------
#將圖片放入db中
#------------------------------------------------------------------------------------------
conn = MongoClient('localhost', 27017) #連結mongodb
#db = conn.NiusNews202010 #create database
db = conn.NiusNews2020_04_12 #create database
#db = db.image
for image in os.listdir(newimgdir): #遍歷圖片目錄集合
    filesname = newimgdir + image
    f = image.split('.') #分割，為了儲存圖片檔案的格式和名稱
    datatmp = open(filesname, 'rb') #類似於建立檔案
    imgput = GridFS(db) #建立寫入流
    insertimg = imgput.put(datatmp, content_type=f[1], filename=f[0]) #將資料寫入，檔案型別和名稱通過前面的分割得到
    datatmp.close()
#建立成功後，會在集合中生成fs.flies和fs.chunks

#%%----------------------------------------------------------------------------------------
#從資料庫取照片
#------------------------------------------------------------------------------------------
gridFS = GridFS(db, collection="fs")
count=0
for grid_out in gridFS.find():
    count+=1
    if count == 5:
        break
    print(count)
    #print(grid_out.filename) #postid
    data = grid_out.read() #獲取圖片資料
    outf = open(str(count)+'.jpg','wb') #建立檔案
    outf.write(data)  #儲存圖片
    outf.close()
# %%

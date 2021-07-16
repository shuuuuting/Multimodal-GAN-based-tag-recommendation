# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:33:58 2017

@author: user
"""
#目的：砍掉所有database，重跑或第一次跑使用，可以確保資料庫為空的
import pymongo

#連接mongo db
conn = pymongo.MongoClient('localhost', 27017)
db = conn.NiusNews2020_04_12
############################################################################################

print('Execute 1-00_Drop_Table.py')
for table in db.collection_names():
	print(table)
	db[table].drop() #將資料表刪除
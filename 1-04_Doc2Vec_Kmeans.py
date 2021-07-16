#%%
import os
from threading import local
import numpy as np
import pandas as pd
import csv
from pymongo import MongoClient

conn = MongoClient('localhost', 27017) #連結mongodb
#db = conn.NiusNews202010 #create database
db = conn.NiusNews2020_04_12 #create database
postsdb = db['post_Jieba_ID50_tagged']
posts = pd.DataFrame(list(postsdb.find({},{"_id": 0,"postid": 1,"content_jieba": 1,"chanel_num":1})))
tags = pd.DataFrame(list(postsdb.find({},{"_id": 0,"postid": 1,"tag": 1})))
posts = posts.set_index('postid')
chanel_tr = posts.loc[list(text_tr_pd.postid)]
chanel_te = posts.loc[list(text_te_pd.postid)]
chanel_tr = chanel_tr.reset_index(drop=True)
chanel_te = chanel_te.reset_index(drop=True)

#%%----------------------------------------------------------------------------------------
#處理成能進行doc2vec的資料
#------------------------------------------------------------------------------------------
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
all_content_train = []
j=0
for em in text_tr_pd['content_jieba'].values:
    all_content_train.append(TaggedDocument(em,[j]))
    j+=1
print('Number of texts processed: ', j)

#%%----------------------------------------------------------------------------------------
#訓練doc2vec
#------------------------------------------------------------------------------------------
d2v_model = Doc2Vec(all_content_train, vector_size=200, min_count=20, workers=4, alpha=0.025, min_alpha=0.001, pretrain_emb='pretrain_emb.txt')
d2v_model.train(all_content_train, total_examples=d2v_model.corpus_count, epochs=20, start_alpha=0.002, end_alpha=-0.016)

vec_sent_0_d2v = d2v_model.infer_vector(text_tr_pd['content_jieba'][0], steps=100, alpha=0.025)
#print(vec_sent_0_d2v)
vec_sent_1_d2v = d2v_model.infer_vector(text_tr_pd['content_jieba'][2], steps=100, alpha=0.025)
#print(vec_sent_1_d2v)
vec_sent_2_d2v = d2v_model.infer_vector(text_tr_pd['content_jieba'][0], steps=100, alpha=0.025)
d2v_similarity_0 = np.dot(vec_sent_0_d2v, vec_sent_1_d2v) / (np.linalg.norm(vec_sent_0_d2v)*np.linalg.norm(vec_sent_1_d2v))
print('similarity of Doc2Vec: ', d2v_similarity_0)
d2v_similarity_1 = np.dot(vec_sent_0_d2v, vec_sent_2_d2v) / (np.linalg.norm(vec_sent_0_d2v)*np.linalg.norm(vec_sent_2_d2v))
print('similarity of Doc2Vec: ', d2v_similarity_1)

#%%----------------------------------------------------------------------------------------
#channel的中心當作Kmeans初始中心
#------------------------------------------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
n_clusters = 15
centroids = []
for i in np.unique(chanel_tr.chanel_num):
    local_tr = chanel_tr[chanel_tr.chanel_num==i]
    local_content = np.array([d2v_model.infer_vector(j, steps=50, alpha=0.025) for j in local_tr.content_jieba])
    centroids.append(np.mean(local_content, axis=0))
centroids = np.array(centroids)

vec = np.array([d2v_model.infer_vector(j, steps=50, alpha=0.025) for j in text_tr_pd.content_jieba])
kmeans_model = KMeans(n_clusters=len(centroids), init=centroids, n_init=10) 
X = kmeans_model.fit(vec)
labels = kmeans_model.labels_.tolist()
l = kmeans_model.fit_predict(vec)
pca = PCA(n_components=2).fit(vec)
datapoint = pca.transform(vec)

import matplotlib.pyplot as plt
#%matplotlib inline
plt.figure
label1 = ['#FFFF00', '#008000', '#0000FF', '#800080','#FFFF22','#F44F00', '#558000', '#0E8125', '#0130AB', '#844080','#F44CC0', '#530A00',
'#FFFF09', '#008009', '#0090FF', '#800090','#FFFF92','#F44F90', '#558090', '#0E8195', '#0190AB', '#844090','#F44CC9', '#530A90']
color = [label1[i] for i in labels]
centroidpoint = pca.transform(centroids)
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=labels, cmap='rainbow')
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.show()

#%%----------------------------------------------------------------------------------------
#kmeans分群
#------------------------------------------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
n_clusters = 15
vec = np.array([d2v_model.infer_vector(j, steps=50, alpha=0.025) for j in text_tr_pd.content_jieba])
#vec = d2v_model.docvecs.vectors_docs 
kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10) 
X = kmeans_model.fit(vec)
labels = kmeans_model.labels_.tolist()
l = kmeans_model.fit_predict(vec)
pca = PCA(n_components=2).fit(vec)
datapoint = pca.transform(vec)

import matplotlib.pyplot as plt
#%matplotlib inline
plt.figure
label1 = ['#FFFF00', '#008000', '#0000FF', '#800080','#FFFF22','#F44F00', '#558000', '#0E8125', '#0130AB', '#844080','#F44CC0', '#530A00',
'#FFFF09', '#008009', '#0090FF', '#800090','#FFFF92','#F44F90', '#558090', '#0E8195', '#0190AB', '#844090','#F44CC9', '#530A90']
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.show()

#%%----------------------------------------------------------------------------------------
#kmeans分群結果
#------------------------------------------------------------------------------------------
cluster_map = pd.DataFrame()
cluster_map['content'] = text_tr_pd.content_jieba
cluster_map['tag'] = list(tag_train)
cluster_map['cluster'] = X.labels_
cluster_map = cluster_map.reset_index(drop=True)
cluster_map['chanel'] = chanel_tr.chanel_num
c0 = cluster_map[cluster_map.cluster == 0]
c1 = cluster_map[cluster_map.cluster == 1]
c2 = cluster_map[cluster_map.cluster == 2]
c3 = cluster_map[cluster_map.cluster == 3]
c4 = cluster_map[cluster_map.cluster == 4]
c5 = cluster_map[cluster_map.cluster == 5]
c6 = cluster_map[cluster_map.cluster == 6]
c7 = cluster_map[cluster_map.cluster == 7]
c8 = cluster_map[cluster_map.cluster == 8]
c9 = cluster_map[cluster_map.cluster == 9] 
c10 = cluster_map[cluster_map.cluster == 10]
c11 = cluster_map[cluster_map.cluster == 11]
c12 = cluster_map[cluster_map.cluster == 12]
c13 = cluster_map[cluster_map.cluster == 13]
c14 = cluster_map[cluster_map.cluster == 14]
'''
c15 = cluster_map[cluster_map.cluster == 15]
c16 = cluster_map[cluster_map.cluster == 16]
c17 = cluster_map[cluster_map.cluster == 17]
c18 = cluster_map[cluster_map.cluster == 18]
c19 = cluster_map[cluster_map.cluster == 19]
'''

#%%----------------------------------------------------------------------------------------
#找出每群最熱門的詞跟tag作為代表
#------------------------------------------------------------------------------------------
from collections import Counter
g = globals()
popularcontent = []
populartag = []
for i in range(n_clusters):
    varname = 'c{}'.format(i)
    popularcontent.append([w for w, wfreq in Counter(sum(g[varname].content,[])).most_common(100)])
    #populartag.append([t for t, tfreq in Counter(sum(g[varname].tag,[])).most_common(10)])

#%%----------------------------------------------------------------------------------------
#階層式分群(agglomerative clustering)
#------------------------------------------------------------------------------------------
import scipy.cluster.hierarchy as shc
plt.title("Dendograms")
dend = shc.dendrogram(shc.linkage(d2v_model.docvecs.vectors_docs , method='ward'))

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
c_pred = cluster.fit_predict(d2v_model.docvecs.vectors_docs)
clf = NearestCentroid()
clf.fit(d2v_model.docvecs.vectors_docs, c_pred)
centroids = clf.centroids_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=cluster.labels_, cmap='rainbow')
plt.show()

#%%----------------------------------------------------------------------------------------
#階層式分群結果
#------------------------------------------------------------------------------------------
cluster_hier = pd.DataFrame()
cluster_hier['content'] = text_tr_pd.content_jieba
cluster_hier['chanel'] = chanel_tr.chanel_num
cluster_hier['tag'] = tag_tr_pd.tag
cluster_hier['cluster'] = cluster.labels_
ch0 = cluster_hier[cluster_hier.cluster == 0]
ch1 = cluster_hier[cluster_hier.cluster == 1]
ch2 = cluster_hier[cluster_hier.cluster == 2]
ch3 = cluster_hier[cluster_hier.cluster == 3]
ch4 = cluster_hier[cluster_hier.cluster == 4]
ch5 = cluster_hier[cluster_hier.cluster == 5]
ch6 = cluster_hier[cluster_hier.cluster == 6]
ch7 = cluster_hier[cluster_hier.cluster == 7]
ch8 = cluster_hier[cluster_hier.cluster == 8]
ch9 = cluster_hier[cluster_hier.cluster == 9]
ch10 = cluster_hier[cluster_hier.cluster == 10]
ch11 = cluster_hier[cluster_hier.cluster == 11]
ch12 = cluster_hier[cluster_hier.cluster == 12]
ch13 = cluster_hier[cluster_hier.cluster == 13]
ch14 = cluster_hier[cluster_hier.cluster == 14]
ch15 = cluster_hier[cluster_hier.cluster == 15]
ch16 = cluster_hier[cluster_hier.cluster == 16]
ch17 = cluster_hier[cluster_hier.cluster == 17]
ch18 = cluster_hier[cluster_hier.cluster == 18]
ch19 = cluster_hier[cluster_hier.cluster == 19]

#%%----------------------------------------------------------------------------------------
#階層分群的中心當作Kmeans初始中心
#------------------------------------------------------------------------------------------
kmeans_model = KMeans(n_clusters=n_clusters, init=centroids, n_init=1, max_iter=100) 
X = kmeans_model.fit(d2v_model.docvecs.vectors_docs)
labels = kmeans_model.labels_.tolist()
l = kmeans_model.fit_predict(d2v_model.docvecs.vectors_docs)

centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=labels, cmap='rainbow')
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.show()

#(實際有用)########################################
#%%----------------------------------------------------------------------------------------
#LDA分主題
#------------------------------------------------------------------------------------------
from gensim import corpora, models
text_data = list(text_tr_pd.content_jieba)
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=15)
doc_topic = []
for e, values in enumerate(lda.inference(corpus)[0]):
    topic_val = 0
    topic_id = 0
    for tid, val in enumerate(values):
        if val > topic_val:
            topic_val = val
            topic_id = tid
    doc_topic.append(topic_id)

text_data = list(text_te_pd.content_jieba)
corpus_te = [dictionary.doc2bow(text) for text in text_data]
doc_topic_te = []
for e, values in enumerate(lda.inference(corpus_te)[0]):
    topic_val = 0
    topic_id = 0
    for tid, val in enumerate(values):
        if val > topic_val:
            topic_val = val
            topic_id = tid
    doc_topic_te.append(topic_id)

#%%----------------------------------------------------------------------------------------
#lda分主題結果
#------------------------------------------------------------------------------------------
topic_belong = pd.DataFrame()
topic_belong['content'] = text_tr_pd.content_jieba
topic_belong['tag'] = list(tag_train)
topic_belong['topic'] = doc_topic
topic_belong = topic_belong.reset_index(drop=True)
topic_belong['chanel'] = chanel_tr.chanel_num
t0 = topic_belong[topic_belong.topic == 0]
t1 = topic_belong[topic_belong.topic == 1]
t2 = topic_belong[topic_belong.topic == 2]
t3 = topic_belong[topic_belong.topic == 3]
t4 = topic_belong[topic_belong.topic == 4]
t5 = topic_belong[topic_belong.topic == 5]
t6 = topic_belong[topic_belong.topic == 6]
t7 = topic_belong[topic_belong.topic == 7]
t8 = topic_belong[topic_belong.topic == 8]
t9 = topic_belong[topic_belong.topic == 9]
t10 = topic_belong[topic_belong.topic == 10]
t11 = topic_belong[topic_belong.topic == 11]
t12 = topic_belong[topic_belong.topic == 12]
t13 = topic_belong[topic_belong.topic == 13]
t14 = topic_belong[topic_belong.topic == 14]
'''
t15 = topic_belong[topic_belong.topic == 15]
t16 = topic_belong[topic_belong.topic == 16]
t17 = topic_belong[topic_belong.topic == 17]
t18 = topic_belong[topic_belong.topic == 18]
t19 = topic_belong[topic_belong.topic == 19]
'''

#%%----------------------------------------------------------------------------------------
#找出每個主題最熱門的詞作為代表
#------------------------------------------------------------------------------------------
from collections import Counter
g = globals()
ldapopularcontent = []
ldapopulartag = []
for i in range(n_clusters):
    varname = 't{}'.format(i)
    ldapopularcontent.append([w for w, wfreq in Counter(sum(g[varname].content,[])).most_common(100)])
    #ldapopulartag.append([t for t, tfreq in Counter(sum(g[varname].tag,[])).most_common(100)])
    
#(實際有用)########################################

#%%----------------------------------------------------------------------------------------
#LDA分群
#------------------------------------------------------------------------------------------
topic_dist = lda.inference(corpus)[0]
t_kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100) 
t_X = t_kmeans_model.fit(topic_dist)
t_labels = t_kmeans_model.labels_.tolist()
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=t_labels, cmap='rainbow')

#%%----------------------------------------------------------------------------------------
#用title進行kmeans分群
#------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
titles = pd.DataFrame(list(postsdb.find({},{"_id": 0,"postid": 1,"title": 1,"tag":1})))
title_train, title_test = train_test_split(titles, test_size=0.2, random_state=11)
title_train = title_train.reset_index(drop=True)
title_test = title_test.reset_index(drop=True)
all_title_train = []
j=0
for em in title_train['title'].values:
    all_title_train.append(TaggedDocument(em,[j]))
    j+=1
print('Number of titles processed: ', j)
title_d2v_model = Doc2Vec(all_title_train, size=100, min_count=100, workers=8, alpha=0.025, min_alpha=0.001)
title_d2v_model.train(all_title_train, total_examples=title_d2v_model.corpus_count, epochs=20, start_alpha=0.002, end_alpha=-0.016)

n_clusters = 20
tkmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100) 
tX = tkmeans_model.fit(title_d2v_model.docvecs.vectors_docs)
tlabels = tkmeans_model.labels_.tolist()
tl = tkmeans_model.fit_predict(title_d2v_model.docvecs.vectors_docs)
tpca = PCA(n_components=2).fit(title_d2v_model.docvecs.vectors_docs)
tdatapoint = tpca.transform(title_d2v_model.docvecs.vectors_docs)

#%%
#直接使用chanel num作為分群(但那也是人為標註的類別...
chanel_popularcontent = []
for i in np.unique(chanel_tr.chanel_num):
    chanel_popularcontent.append([w for w, wfreq in Counter(sum(chanel_tr[chanel_tr.chanel_num==i].content_jieba,[])).most_common(50)])
#Counter(chanel_tr.chanel_num) #check channel個別文章數量


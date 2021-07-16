#%% save 
tr_arr_reshaped = image_train.reshape(image_train.shape[0], -1)
np.savetxt(datadir+"image_train.txt", tr_arr_reshaped)
te_arr_reshaped = image_test.reshape(image_test.shape[0], -1)
np.savetxt(datadir+"image_test.txt", te_arr_reshaped)

tr_arr_reshaped = embedding_train.reshape(embedding_train.shape[0], -1)
np.savetxt(datadir+"embedding_train.txt", tr_arr_reshaped)
te_arr_reshaped = embedding_test.reshape(embedding_test.shape[0], -1)
np.savetxt(datadir+"embedding_test.txt", te_arr_reshaped)

tr_arr_reshaped = embedding_title_train.reshape(embedding_title_train.shape[0], -1)
np.savetxt(datadir+"embedding_title_train.txt", tr_arr_reshaped)
te_arr_reshaped = embedding_title_test.reshape(embedding_title_test.shape[0], -1)
np.savetxt(datadir+"embedding_title_test.txt", te_arr_reshaped)

#%%reload arr
import numpy as np
datadir = 'C:/Users/shuting/Desktop/論文/data/'

embedding_size = 200
loaded_arr = np.loadtxt(datadir+"image_train.txt")
image_train = loaded_arr.reshape(
    loaded_arr.shape[0], 7, 7, 512)
loaded_arr = np.loadtxt(datadir+"image_test.txt")
image_test = loaded_arr.reshape(
    loaded_arr.shape[0], 7, 7, 512)

loaded_arr = np.loadtxt(datadir+"embedding_train.txt")
embedding_train = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)
loaded_arr = np.loadtxt(datadir+"embedding_test.txt")
embedding_test = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)

loaded_arr = np.loadtxt(datadir+"embedding_title_train.txt")
embedding_title_train = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)
loaded_arr = np.loadtxt(datadir+"embedding_title_test.txt")
embedding_title_test = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)

#%%
loaded_arr = np.loadtxt(datadir+"lda_train.txt")
lda_train = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)
loaded_arr = np.loadtxt(datadir+"lda_test.txt")
lda_test = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size) 
    
loaded_arr = np.loadtxt(datadir+"lda_tr_arr.txt")
lda_tr_arr = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)
loaded_arr = np.loadtxt(datadir+"lda_te_arr.txt")
lda_te_arr = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)

loaded_arr = np.loadtxt(datadir+"lda_title_tr_arr.txt")
lda_title_tr_arr = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)
loaded_arr = np.loadtxt(datadir+"lda_title_te_arr.txt")
lda_title_te_arr = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)
# %%

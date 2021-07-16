#%%
import math
from ml_metrics import mapk
def evaluation(y_true, y_pred, top_K):
    acc_count = 0
    precision_K = []
    recall_K = []
    f1_K = []
    ndcg_K = []
    true_sum = 0

    for i in range(y_pred.shape[0]):
        top_indices = y_pred[i].argsort()[-top_K:][::-1] #每篇文章排名前10可能的tag index
        true_num = np.sum(y_true[i, :])
        true_sum += true_num
        dcg = 0
        idcg = 0
        idcgCount = true_num
        j = 0
        for item in top_indices:
            if y_true[i, item] == 1:
                dcg += 1.0/math.log2(j + 2)
            if idcgCount > 0:
                idcg += 1.0/math.log2(j + 2)
                idcgCount = idcgCount-1
            j += 1
        if(idcg != 0):
            ndcg_K.append(dcg/idcg)

        if np.sum(y_true[i, top_indices]) >= 1: #代表預測出來要推薦的tag有hit到真實tag
            acc_count += 1
        p = np.sum(y_true[i, top_indices])/top_K
        r = np.sum(y_true[i, top_indices])/np.sum(y_true[i, :])
        precision_K.append(p)
        recall_K.append(r)
        if p != 0 or r != 0:
            f1_K.append(2 * p * r / (p + r))
        else:
            f1_K.append(0)
    acc_K = acc_count * 1.0 / y_pred.shape[0]
    trueidlist = [np.where(i==1)[0].tolist() for i in y_true]
    predidorder = np.flip(y_pred.argsort()[:,-top_K:],axis=1)
    map_K = mapk(trueidlist, predidorder, top_K)

    return acc_K, np.mean(np.array(precision_K)), np.mean(np.array(recall_K)), np.mean(np.array(f1_K)), np.mean(np.array(ndcg_K)), map_K
# %%

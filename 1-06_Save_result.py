#%%
from openpyxl import Workbook
'''
wb = Workbook()
wb.remove(wb.active)
ws = wb.create_sheet(title=method)
'''
def save_result(acc_K, precision_K, recall_K, f1_K, ndcg_K, map_K):
    ws.append([acc_K, precision_K, recall_K, f1_K, ndcg_K, map_K])
    return

# %%

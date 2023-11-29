from numpy import *
# from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np


ranking_df = pd.read_csv("results.txt", header=None, delimiter = '\t')
ranking_df.columns = ['TF', 'Target','Score']
print(ranking_df)
# 过滤目标基因标识符小于100的行, 因为得到的ranking.txt数据里面有一些 error
valid_ranking_df = ranking_df[ranking_df['Target'] >= 100]

# 根据分数选择前N个调控关系
top_n = 100  # 例如，选择前100个
top_ranking_df = valid_ranking_df.nlargest(top_n, 'Score')

# 按照TF列排序
sorted_df = top_ranking_df.sort_values(by='TF')

sorted_df.to_csv("100_mr_result_sort.csv", index=False)
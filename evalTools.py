import pandas as pd
from numpy import *
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# run this file before evaluation matrix
def sortResults(generatedMatrix, method, top_n, testSetNum):
    ranking_df = pd.read_csv(generatedMatrix, header=None, delimiter = '\t')
    ranking_df.columns = ['TF', 'Target','Score']
    print(ranking_df)
    # 过滤目标基因标识符小于100的行, 因为得到的ranking.txt数据里面有一些 error
    valid_ranking_df = ranking_df[ranking_df['Target'] >= 100]

    # 根据分数选择前N个(top_n)调控关系
    top_ranking_df = valid_ranking_df.nlargest(top_n, 'Score')

    # 按照TF列排序
    sorted_df = top_ranking_df.sort_values(by='TF')
    # Select set num from [5 50 100]
    sorted_df.to_csv("sorted_data/{0}_mr_result_sorted_top_{1}_method_{2}.csv".format(testSetNum, top_n, method), index=False)
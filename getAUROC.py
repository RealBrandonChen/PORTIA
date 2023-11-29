from numpy import *
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


'''
考虑的Evaluation index:
1. AUROC
2. AUPRC?
3. Accuracy
4. Precision & Recall
5. F1 Score
6. Confusion Matrix
'''

if __name__ == '__main__':

    ####################################################################################################################
    # Get the truth label
    true_network = pd.read_csv('bipartite_GRN.csv', header=None, names=['TF', 'Target'])
    # 初始化一个二维的真实标签数组
    num_tfs = 100  # 转录因子的数量
    num_targets = 100  # 目标基因的数量
    true_labels = np.zeros((num_tfs, num_targets))

    # 填充真实标签矩阵
    for _, row in true_network.iterrows():
        tf_idx = row['TF']
        target_idx = row['Target'] - 100  # 减去100，因为目标基因的索引从100开始
        true_labels[tf_idx, target_idx] = 1

    ####################################################################################################################
    # Get the predict label

    # valid_ranking_df = pd.read_csv("100_mr_sort_result.csv")
    # 读取 ranking.txt 文件
    ranking_df = pd.read_csv("results.txt", header=None, delimiter = '\t')
    ranking_df.columns = ['TF', 'Target','Score']


    # 过滤目标基因标识符小于100的行, 因为得到的ranking.txt数据里面有一些 error
    valid_ranking_df = ranking_df[ranking_df['Target'] >= 100]


    # 初始化预测分数数组，大小与 true_labels 相同
    predicted_scores_reordered = np.zeros(true_labels.shape)

    # 填充重排序的预测分数
    for _, row in valid_ranking_df.iterrows():
        tf_idx = int(row['TF'])  # 确保 tf_idx 是整数
        target_idx = int(row['Target'] - 100)  # 确保 target_idx 是整数
        score = row['Score']
        predicted_scores_reordered[tf_idx, target_idx] = score

    # # 展平数组以计算 总体的 AUROC
    true_labels_flattened = true_labels.flatten()
    predicted_scores_flattened = predicted_scores_reordered.flatten()

    # 计算 AUROC, 需要flatten数组
    auc = roc_auc_score(true_labels_flattened, predicted_scores_flattened)
    print("AUROC:", auc)

    ####################################################################################################################
    # Get AUROC
    # 存储每个基因的AUROC
    gene_aurocs = []

    for target_idx in range(num_targets):
        # 提取与当前目标基因相关的真实标签和预测分数
        true_labels_gene = true_labels[:, target_idx]
        predicted_scores_gene = predicted_scores_reordered[:, target_idx]

        # 计算并存储当前基因的AUROC
        auroc = roc_auc_score(true_labels_gene, predicted_scores_gene)
        gene_aurocs.append(auroc)

    # 创建DataFrame并保存为CSV文件
    auroc_df = pd.DataFrame(gene_aurocs, columns=['AUROC'])
    csv_file_path = "100_mr_gene_aurocs.csv"  # 您可以根据需要修改文件路径
    auroc_df.to_csv(csv_file_path, index=False, header=False)

# ########################################################################################################################
#     # 绘制AUROC条形图
#     def plot_auroc_bar_chart(aurocs):
#         plt.figure(figsize=(15, 10))
#         plt.bar(range(len(aurocs)), aurocs)
#         plt.xlabel('Target Gene Index')
#         plt.ylabel('AUROC')
#         plt.title('AUROC for Each Target Gene')
#         plt.show()
#
#
#     # 绘制基因关系热图
#     def plot_gene_relation_heatmap(predicted_scores):
#         plt.figure(figsize=(20, 15))
#         sns.heatmap(predicted_scores, cmap='viridis')
#         plt.xlabel('Target Gene Index')
#         plt.ylabel('TF Index')
#         plt.title('Predicted Gene Regulatory Relationships')
#         plt.show()
#
#
#     # 绘制条形图
#     plot_auroc_bar_chart(gene_aurocs)
#
#     # 绘制热图
#     plot_gene_relation_heatmap(predicted_scores_reordered)
#
# ########################################################################################################################
# # 绘制散点图以展示基因与TF之间的关系
# # 绘制散点图来展示TF和目标基因表达水平之间的关系
# def plot_tf_gene_scatter(true_network, predicted_scores_reordered):
#     # 准备画布
#     fig, axes = plt.subplots(10, 10, figsize=(20, 20), sharex=True, sharey=True)
#     axes = axes.flatten()
#
#     # 对每对TF和目标基因绘制散点图
#     for idx, ax in enumerate(axes):
#         if idx >= num_tfs:
#             break
#         # 筛选当前TF对所有基因的预测分数
#         tf_scores = predicted_scores_reordered[idx, :]
#         # 绘制散点图
#         ax.scatter(range(num_targets), tf_scores, alpha=0.6)
#         ax.set_title(f'TF_{idx}')
#         ax.grid(True)
#
#     # 设置画布的标题和标签
#     plt.suptitle('Scatter Plots of TFs and Target Genes Predicted Scores')
#     fig.text(0.5, 0.04, 'Target Genes', ha='center', va='center')
#     fig.text(0.06, 0.5, 'Predicted Scores', ha='center', va='center', rotation='vertical')
#
#     # 调整布局并显示图形
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.95)
#     plt.show()
#
# # 绘制散点图
# plot_tf_gene_scatter(true_network, predicted_scores_reordered)


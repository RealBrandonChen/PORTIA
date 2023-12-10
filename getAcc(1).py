import pandas as pd
import os

def calculate_accuracy(predicted_file, true_file):

    # 读取预测结果和真实数据
    predicted_df = pd.read_csv(predicted_file)
    true_df = pd.read_csv(true_file)

    # 转换为集合形式
    predicted_pairs = set(predicted_df.apply(lambda row: (row["TF"], row["Target"]), axis=1))
    true_pairs = set(true_df.apply(lambda row: (row.iloc[0], row.iloc[1]), axis=1))

    # 计算True Positives, False Positives, 和 False Negatives
    true_positives = len(predicted_pairs.intersection(true_pairs))
    false_positives = len(predicted_pairs - true_pairs)
    false_negatives = len(true_pairs - predicted_pairs)

    # 计算 Precision, Recall 和 F1 Score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 计算正确预测的对数
    correct_predictions = predicted_pairs.intersection(true_pairs)
    # 计算准确率
    accuracy = len(correct_predictions) / len(predicted_pairs)
    return accuracy, precision, recall, f1_score

true_file_5 = '5_mr_50_cond/bipartite_GRN.csv'  
true_file_40 = '40_mr_50_cond/bipartite_GRN.csv' 
true_file_100 = '100_mr_50_cond/bipartite_GRN.csv' # 替换为真实数据文件路径

df_5 = pd.DataFrame(columns = ['File','Accuracy', 'Precision','Recall','F1 Score'])
# 调用函数计算准确率
for f in os.listdir("sorted_data_5"):
    predicted_file = os.path.join('sorted_data_5',f)  # 替换为预测结果文件路径
    accuracy, precision, recall, f1_score = calculate_accuracy(predicted_file, true_file=true_file_5)
    df_5.loc[len(df_5.index)] = [f, accuracy, precision, recall, f1_score] 
    df_5.to_csv("Accuracy/Accuracy_5.csv", index=False)
    
df_40 = pd.DataFrame(columns = ['File','Accuracy', 'Precision','Recall','F1 Score'])
# 调用函数计算准确率
for f in os.listdir("sorted_data_40"):
    predicted_file = os.path.join('sorted_data_40',f)  # 替换为预测结果文件路径
    accuracy, precision, recall, f1_score = calculate_accuracy(predicted_file, true_file=true_file_40)
    df_40.loc[len(df_40.index)] = [f, accuracy, precision, recall, f1_score] 
    df_40.to_csv("Accuracy/Accuracy_40.csv", index=False)
    
df_100 = pd.DataFrame(columns = ['File','Accuracy', 'Precision','Recall','F1 Score'])
# 调用函数计算准确率
for f in os.listdir("sorted_data_100"):
    predicted_file = os.path.join('sorted_data_100',f)  # 替换为预测结果文件路径
    accuracy, precision, recall, f1_score = calculate_accuracy(predicted_file, true_file=true_file_100)
    df_100.loc[len(df_100.index)] = [f, accuracy, precision, recall, f1_score] 
    df_100.to_csv("Accuracy/Accuracy_100.csv", index=False)
import pandas as pd

def calculate_accuracy(predicted_file, true_file):

    # 读取预测结果和真实数据
    predicted_df = pd.read_csv(predicted_file)
    true_df = pd.read_csv(true_file)

    # 转换为集合形式
    predicted_pairs = set(predicted_df.apply(lambda row: (row["TF"], row["Target"]), axis=1))
    print("Predicted pairs: " + str(predicted_pairs))
    true_pairs = set(true_df.apply(lambda row: (row.iloc[0], row.iloc[1]), axis=1))
    print("True pairs: " + str(true_pairs))

    # 计算正确预测的对数
    correct_predictions = predicted_pairs.intersection(true_pairs)
    print("Correct predictions: " + str(correct_predictions))
    # 计算准确率
    accuracy = len(correct_predictions) / len(predicted_pairs)
    return accuracy

# 调用函数计算准确率
predicted_file = '100_mr_result_sort.csv'  # 替换为预测结果文件路径
true_file = '100_mr_50_cond/bipartite_GRN.csv'                       # 替换为真实数据文件路径
accuracy = calculate_accuracy(predicted_file, true_file)
print("Accuracy:", accuracy)
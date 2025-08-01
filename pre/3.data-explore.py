import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def normalize_table(table):
    """
    将数据归一化到 [0, 1] 区间
    :param table: DataFrame，需要归一化的表格
    :return: DataFrame，归一化后的表格
    """
    normalized_table = table.copy()
    # 检查是否有 author_name 列
    if 'author_name' not in normalized_table.columns:
        raise ValueError("The input CSV file must contain an 'author_name' column.")

    # 添加 author_id 列，从 0 开始编号
    normalized_table.insert(0, 'author_id', range(len(normalized_table['author_name'])))

    # 删除 author_name 列
    normalized_table.drop(columns=['author_name'], inplace=True)

    # 遍历每一列，检查是否是数值列
    for column in normalized_table.columns[1:]:  # 跳过第一列（非数值列）
        if pd.api.types.is_numeric_dtype(normalized_table[column]):
            min_val = normalized_table[column].min()
            max_val = normalized_table[column].max()
            if max_val > min_val:  # 避免分母为 0
                # 按照 [-1, 1] 到 [0, 1] 的规则归一化
                normalized_table[column] = (normalized_table[column] + 1) / 2
            else:
                normalized_table[column] = 0.5  # 所有值相同，直接归一化为 0.5
    return normalized_table

# 示例主流程
# if __name__ == "__main__":
#     # 假设清洗后的数据已经保存为 CSV 文件
#     cleaned_table_file = "author_sentiment_table.csv"
#     output_normalized_file = "../cc/normalized_author_sentiment_table.csv"
#
#     # 加载清洗后的数据表
#     table = pd.read_csv(cleaned_table_file)
#
#     # 对数据进行归一化
#     normalized_table = normalize_table(table)
#
#     # 保存归一化后的数据
#     normalized_table.to_csv(output_normalized_file, index=False)
#     print(f"归一化后的数据已保存到 {output_normalized_file}")


def analyze_and_visualize_last_column(table, output_plot_file):
    """
    对清洗后的最后一列数据进行分析和可视化
    :param table: DataFrame，已清洗的数据表
    :param output_plot_file: 保存可视化结果的文件路径
    """
    # 获取最后一列
    last_column = table.iloc[:, -1]

    # 按 -1 到 1 以 0.1 为间隔划分为 20 类
    bins = np.linspace(0, 1, 6)  # 生成分段 [-1, -0.9, ..., 1.0]
    bin_labels = [f"{bins[i]:.1f} ~ {bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    table['categories'] = pd.cut(last_column, bins=bins, labels=bin_labels, include_lowest=True)

    # 统计各类数量
    category_counts = table['categories'].value_counts().sort_index()

    # 打印统计结果
    print("分类统计结果：")
    print(category_counts)

    # 可视化
    plt.figure(figsize=(12, 6))
    category_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Distribution of Sentiment Scores", fontsize=16)
    plt.xlabel("Sentiment Score Range", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()

    # 保存图像
    plt.savefig(output_plot_file)
    print(f"可视化结果已保存到 {output_plot_file}")


# 示例主流程
if __name__ == "__main__":
    # 假设清洗后的数据已经保存为 CSV 文件
    cleaned_table_file = "../ruua/author_sentiment_table.csv"
    output_normalized_file = "../ruua/normalized_author_sentiment_table.csv"
    output_plot_file = "../ruua/sentiment_distribution.png"

    # 加载清洗后的数据表
    table = pd.read_csv(cleaned_table_file)

    # 对数据进行归一化
    normalized_table = normalize_table(table)

    # 保存归一化后的数据
    normalized_table.to_csv(output_normalized_file, index=False)
    print(f"归一化后的数据已保存到 {output_normalized_file}")

    # 对最后一列数据进行分析和可视化
    df = pd.read_csv(output_normalized_file)
    analyze_and_visualize_last_column(df, output_plot_file)

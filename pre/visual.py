import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_individual_opinions(data_path, sample_size=100):
    """
    Visualize individual opinion changes over time for a sample of users.

    Parameters:
    - data_path: Path to the CSV file containing the data.
    - sample_size: Number of users to sample for visualization.
    """
    # Load the dataset
    df = pd.read_csv(data_path)

    # Randomly sample a subset of users
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Extract author IDs and time columns
    author_ids = df['author_id']
    time_columns = df.columns[1:]  # All columns except the first one

    # Plot each author's opinion over time
    plt.figure(figsize=(14, 10))  # Increase figure size

    for i in range(len(df)):
        opinions = df.iloc[i][time_columns].values  # Use iloc for positional indexing
        plt.plot(time_columns, opinions, label=f"Author {df.iloc[i]['author_id']}", alpha=0.7)

    # Customize the plot
    plt.xlabel("Time")
    plt.ylabel("Opinion")
    plt.title("Individual Opinion Changes Over Time (Sampled)")
    plt.xticks(rotation=45)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')  # Adjust legend position
    plt.subplots_adjust(right=0.8)  # Create space for the legend on the right
    plt.tight_layout()

    # Show the plot
    plt.show()

def analyze_and_visualize_last_column(table):
    """
    对清洗后的最后一列数据进行分析和可视化
    :param table: DataFrame，已清洗的数据表
    :param output_plot_file: 保存可视化结果的文件路径
    """
    # 获取最后一列
    last_column = table.iloc[:, 22]

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

    plt.show()

    # 保存图像
    # plt.savefig(output_plot_file)
    # print(f"可视化结果已保存到 {output_plot_file}")


# Example usage
data_path = "../cc/normalized_author_sentiment_table.csv"  # Replace with your CSV file path

# plot_individual_opinions(data_path, sample_size=100)

# 对最后一列数据进行分析和可视化
df = pd.read_csv(data_path)
analyze_and_visualize_last_column(df)
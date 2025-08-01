import pandas as pd
import numpy as np
from datetime import datetime
from transformers import pipeline
from tqdm import tqdm
import torch

# 1. 构建表格
def construct_author_table(data, selected_authors, start_date, end_date):
    """构建表格，按月聚合文本
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS")  # MS: Month Start
    columns = ["author_name"] + [date.strftime("%Y-%m") for date in date_range]

    table = pd.DataFrame(columns=columns)
    table["author_name"] = selected_authors

    for author in selected_authors:
        author_data = data[data["author_name"] == author]
        for _, row in author_data.iterrows():
            month = row["created_time"].strftime("%Y-%m")
            if month in table.columns and not pd.isna(row["self_text"]):
                current_text = table.loc[table["author_name"] == author, month].values[0]
                new_text = row["self_text"]

                # 转换为字符串并处理 NaN 值
                current_text = "" if pd.isna(current_text) else str(current_text)
                new_text = str(new_text)

                # 拼接文本
                table.loc[table["author_name"] == author, month] = current_text + " " + new_text
    return table

# 2. 使用 Hugging Face 模型进行情感分析
def sentiment_analysis_huggingface(pipeline, text):
    """使用 Hugging Face pipeline 进行情感分析
    """
    if pd.isna(text):
        return np.nan
    try:
        result = pipeline(text[:512])  # 截断到 512 字符以支持模型输入
        label = result[0]['label']
        score = result[0]['score']
        return round(score, 3) if label == "POSITIVE" else round(-score, 3)
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return np.nan

def apply_sentiment_analysis(table, sentiment_analyzer):
    """遍历表格并进行情感分析
    """
    for column in tqdm(table.columns[1:], desc="Processing Sentiments"):
        table[column] = table[column].apply(lambda text: sentiment_analysis_huggingface(sentiment_analyzer, text))
    return table

# 3. 填充缺失数据
def fill_missing_data(table):
    """填充缺失数据
    """
    for index, row in table.iterrows():
        for col_idx in range(1, len(row)):
            if pd.isna(row[col_idx]):
                if col_idx == 1:
                    table.iat[index, col_idx] = 0
                else:
                    table.iat[index, col_idx] = table.iat[index, col_idx - 1]
    return table

# 4. 保存数据
def save_table(table, output_file):
    """保存数据到文件
    """
    table.to_csv(output_file, index=False)
    print(f"数据已保存到 {output_file}")

# 主流程
if __name__ == "__main__":
    # 配置参数
    input_file = "../ruua/reddit_opinion_ru_ua.csv"
    selected_authors_file = "../ruua/selected_users.txt"
    output_file = "../ruua/author_sentiment_table.csv"
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 11, 30)

    # 加载数据
    data = pd.read_csv(input_file, encoding="ISO-8859-1", low_memory=False)
    data["created_time"] = pd.to_datetime(data["created_time"], errors="coerce")

    # 加载筛选的用户名单
    with open(selected_authors_file, "r") as f:
        selected_authors = [line.strip() for line in f]

    # 构建表格
    table = construct_author_table(data, selected_authors, start_date, end_date)

    # 检查 GPU 支持
    device = 0 if torch.cuda.is_available() else -1

    # 加载 Hugging Face 的情感分析模型
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0f",
        device=device
    )

    # 使用情感分析算法打分
    table = apply_sentiment_analysis(table, sentiment_analyzer)

    # 填充数据
    table = fill_missing_data(table)

    # 保存结果
    save_table(table, output_file)

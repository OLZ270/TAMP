import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('climate/author_sentiment_table.csv')

# 4. 清洗数据
def clean_data(table, threshold=0.2):
    """从第二列开始，删除非零数据总数小于阈值的列"""
    total_rows = len(table)
    for column in table.columns[1:]:  # 跳过第一列
        non_zero_count = table[column].apply(lambda x: x != 0 and not pd.isna(x)).sum()
        if non_zero_count / total_rows < threshold:
            table.drop(columns=[column], inplace=True)
        else:
            break
    return table

# 5. 保存数据
def save_table(table, output_file):
    """保存数据到文件"""
    table.to_csv(output_file, index=False)
    print(f"数据已保存到 {output_file}")

data = clean_data(df, threshold=0.2)

output_file = "climate/author_sentiment_cleaned.csv"

save_table(data, output_file)
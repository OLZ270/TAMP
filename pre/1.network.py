import pandas as pd
import scipy.sparse as sp
from datetime import datetime

# 1. 加载数据

# 转换时间范围
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 11, 30)

# 只加载必要的列：评论时间（created_time）、用户名（author_name）、帖子ID（post_id）、子版块（subreddit）
data = pd.read_csv('../ruua/reddit_opinion_ru_ua.csv', encoding="ISO-8859-1",
                   usecols=["created_time", "author_name", "post_id", "subreddit"])

# 转换 created_time 列为 datetime 类型
data['created_time'] = pd.to_datetime(data['created_time'], errors='coerce')

# 筛选时间范围内的评论
filtered_data = data[(data['created_time'] >= start_date) & (data['created_time'] <= end_date)]

# 统计每个用户的评论数量
author_counts = filtered_data['author_name'].value_counts()

# 筛选评论数不少于10的用户
selected_authors = author_counts[author_counts >= 20].index.tolist()

# 输出筛选的用户名单
print("筛选的用户长度:", len(selected_authors))

# 2. 构建筛选用户之间的网络

# 只保留 selected_authors 中的用户
filtered_data = filtered_data[filtered_data['author_name'].isin(selected_authors)]

# 创建映射，便于后续处理
author_to_index = {author: idx for idx, author in enumerate(selected_authors)}

# 创建稀疏矩阵的行、列索引和权重数据
rows, cols = [], []
data_values = []

# 按 post_id 分组构建边
for post_id, group in filtered_data.groupby('post_id'):
    users = group['author_name'].unique()
    users = [user for user in users if user in selected_authors]  # 仅保留 selected_authors 的用户

    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            u_idx = author_to_index[users[i]]
            v_idx = author_to_index[users[j]]
            # 将边的信息添加到稀疏矩阵数据中
            rows.append(u_idx)
            cols.append(v_idx)
            data_values.append(1)  # 初始权重为1
            rows.append(v_idx)
            cols.append(u_idx)
            data_values.append(1)  # 无向图，重复添加

# 创建稀疏邻接矩阵
adj_matrix = sp.coo_matrix((data_values, (rows, cols)), shape=(len(selected_authors), len(selected_authors)))
#
# # 按 subreddit 分组构建时间相关的边
# for subreddit, group in data.groupby("subreddit"):
#     group = group[group['author_name'].isin(selected_authors)]  # 限制为 selected_authors
#     group = group.sort_values(by="created_time")
#     authors = group["author_name"].tolist()
#     times = group["created_time"].tolist()
#
#     for i in range(len(authors)):
#         for j in range(i + 1, len(authors)):
#             time_diff = (times[j] - times[i]).total_seconds()
#             if time_diff > 3600:  # 超过 1 小时则跳过
#                 break
#             u_idx = author_to_index[authors[i]]
#             v_idx = author_to_index[authors[j]]
#             rows.append(u_idx)
#             cols.append(v_idx)
#             data_values.append(1)
#             rows.append(v_idx)
#             cols.append(u_idx)
#             data_values.append(1)
#
# # 将时间相关的边添加到稀疏矩阵中
# adj_matrix = sp.coo_matrix((data_values, (rows, cols)), shape=(len(selected_authors), len(selected_authors)))

# 打印图的基本信息
num_nodes = adj_matrix.shape[0]
num_edges = len(rows) // 2  # 边的数量是每个边出现两次（无向图）

# 打印图的基本信息
print(f"图中节点数量: {num_nodes}")
print(f"图中边的数量: {num_edges}")
print(f"筛选的用户数量: {len(selected_authors)}")

# 3. 保存网络和筛选后的用户名单

# 保存用户名单
user_list_file = "../ruua/selected_users.txt"
with open(user_list_file, "w") as f:
    for user in selected_authors:
        f.write(f"{user}\n")
print(f"筛选的用户名单已保存到 {user_list_file}")

# 保存稀疏邻接矩阵为文件
network_file = "../ruua/user_network.npz"
sp.save_npz(network_file, adj_matrix)
print(f"用户网络已保存为稀疏矩阵格式到 {network_file}")

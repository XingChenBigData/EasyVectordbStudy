"""
FAISS基础示例：随机向量精确检索（IndexFlatL2）
流程：生成数据 → 创建索引 → 添加向量 → 执行检索 → 解析结果
"""
import faiss
import numpy as np

# -------------------------- 1. 配置参数与生成数据 --------------------------
# 向量配置
dimension = 128  # 向量维度（模拟图像/文本特征向量）
db_size = 10000  # 数据库向量数量（1万个）
query_size = 5  # 查询向量数量（5个）
k = 10  # 每个查询返回Top-10相似结果

# 生成数据库向量（float32类型，形状：(db_size, dimension)）
np.random.seed(42)  # 固定随机种子，保证结果可复现
db_vectors = np.random.random((db_size, dimension)).astype('float32')

# 生成查询向量（形状：(query_size, dimension)）
query_vectors = np.random.random((query_size, dimension)).astype('float32')

# -------------------------- 2. 创建索引 --------------------------
# 使用IndexFlatL2（基于L2距离的精确检索索引）
# 构造函数参数：向量维度
index = faiss.IndexFlatL2(dimension)

# 查看索引状态（是否已训练，FAISS中精确索引无需训练，默认is_trained=True）
print("索引是否已训练：", index.is_trained)  # 输出：True
print("索引初始向量数量：", index.ntotal)  # 输出：0（未添加向量）

# -------------------------- 3. 向索引添加向量 --------------------------
# 使用add()方法添加数据库向量
index.add(db_vectors)

# 查看添加后的索引状态
print("添加后索引向量数量：", index.ntotal)  # 输出：10000

# -------------------------- 4. 执行检索 --------------------------
# 使用search()方法执行检索，参数：查询向量、返回结果数
distances, indices = index.search(query_vectors, k)

# -------------------------- 5. 解析检索结果 --------------------------
print("\n" + "="*50)
print("检索结果解析（共{}个查询，每个返回Top-{}）".format(query_size, k))
print("="*50)

for i in range(query_size):
    print("\n查询向量{}:".format(i+1))
    print("  最相似的{}个向量ID：{}".format(k, indices[i]))
    print("  对应的L2距离：{}".format(np.round(distances[i], 4)))  # 保留4位小数

# -------------------------- 6. 索引的其他常用操作 --------------------------
# 1. 保存索引到磁盘
faiss.write_index(index, "faiss_flatl2_index.index")
print("\n索引已保存到：faiss_flatl2_index.index")

# 2. 从磁盘加载索引
loaded_index = faiss.read_index("faiss_flatl2_index.index")
print("加载的索引向量数量：", loaded_index.ntotal)  # 输出：10000

# 3. 清空索引
loaded_index.reset()
print("清空后索引向量数量：", loaded_index.ntotal)  # 输出：0
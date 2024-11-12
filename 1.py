import faiss
import numpy as np

# 假设有10000个图像的特征，每个特征是128维的
d = 128
nb = 10000
nq = 10  # 查询数量

# 随机生成一些数据作为示例
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# 创建一个IVF索引
nlist = 100  # 分桶数量
k = 4  # 检索的近邻数

quantizer = faiss.IndexFlatL2(d)  # 使用L2距离的平面索引
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

# 训练索引
index.train(xb)
index.add(xb)

# 搜索最近邻
D, I = index.search(xq, k)
print("最近邻的索引：", I)
print("最近邻的距离：", D)

import os
import numpy as np

# 获取当前脚本所在目录
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_X_PATH = os.path.join(DATA_DIR, "train_X.npy")
TRAIN_Y_PATH = os.path.join(DATA_DIR, "train_y.npy")

# 同理可以加载测试集
TEST_X_PATH = os.path.join(DATA_DIR, "test_X.npy")
TEST_Y_PATH = os.path.join(DATA_DIR, "test_y.npy")

# 加载数据
X_train = np.load(TRAIN_X_PATH)
y_train = np.load(TRAIN_Y_PATH)

# 提取第 6 条样本（索引 5）
sample_X = X_train[5]  # shape: (128, 9)
sample_y = y_train[5]  # 标签

# print("第6条样本特征 shape:", sample_X.shape)
# print("第6条样本标签:", sample_y)
# print("第6条样本特征矩阵:")
# print(sample_X)

# print shape
print("训练集 shape:", X_train.shape)
print("训练集标签 shape:", y_train.shape)
print("第6条样本特征 shape:", sample_X.shape)
print("第6条样本标签:", sample_y)
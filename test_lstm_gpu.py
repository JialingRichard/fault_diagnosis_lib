import torch
import torch.nn as nn

# 检查 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ======= 测试 CNN =======
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)  # 输入 32x32

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleCNN().to(device)
x = torch.randn(4, 3, 32, 32, device=device)

# ======= 测试矩阵乘法 =======
try:
    A = torch.randn(1024, 1024, device=device)
    B = torch.randn(1024, 1024, device=device)
    C = torch.matmul(A, B)
    print("Matrix multiplication successful! Result shape:", C.shape)
except Exception as e:
    print("Matrix multiplication failed!")
    print(e)



try:
    y = model(x)
    print("CNN forward pass successful! Output shape:", y.shape)
except Exception as e:
    print("CNN forward pass failed!")
    print(e)


import torch

print("PyTorch version:", torch.__version__)
print("ROCm version:", torch.version.cuda)
print("HIP version:", torch.version.hip)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    print("Device arch:", torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor)

# 尝试查看是否支持卷积
try:
    x = torch.randn(1, 3, 8, 8, device="cuda")
    w = torch.randn(4, 3, 3, 3, device="cuda")
    y = torch.nn.functional.conv2d(x, w)
    print("✅ Conv2d 成功")
except Exception as e:
    print("❌ Conv2d 失败:", str(e))
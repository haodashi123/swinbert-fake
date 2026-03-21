import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device Name:     {torch.cuda.get_device_name(0)}")
    print("✅ 恭喜！环境配置成功，RTX 3050 已就绪！")
else:
    print("❌ 依然失败。请检查显卡驱动。")
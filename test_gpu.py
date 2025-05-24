# test_gpu.py - Verify RTX 4060 Ti is detected
import torch

print("🎮 GPU Test for HOI4 AI")
print("=" * 40)

# Check if CUDA is available
if torch.cuda.is_available():
    print("✅ CUDA is available!")
    print(f"🎯 GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print(f"🚀 CUDA Version: {torch.version.cuda}")

    # Quick speed test
    print("\n⚡ Speed Test:")

    # CPU test
    cpu_tensor = torch.randn(1000, 1000)
    import time

    start = time.time()
    for _ in range(100):
        _ = torch.matmul(cpu_tensor, cpu_tensor)
    cpu_time = time.time() - start

    # GPU test
    gpu_tensor = torch.randn(1000, 1000).cuda()
    torch.cuda.synchronize()  # Wait for GPU
    start = time.time()
    for _ in range(100):
        _ = torch.matmul(gpu_tensor, gpu_tensor)
    torch.cuda.synchronize()
    gpu_time = time.time() - start

    print(f"CPU Time: {cpu_time:.2f} seconds")
    print(f"GPU Time: {gpu_time:.2f} seconds")
    print(f"🏎️ GPU is {cpu_time / gpu_time:.1f}x faster!")

else:
    print("❌ CUDA not available")
    print("Running on CPU only")

print("\n✨ Ready to train HOI4 AI!")
import cv2 as cv
import torch

# OpenCV CUDA check
cuda_count = cv.cuda.getCudaEnabledDeviceCount()
print(f"[OpenCV] CUDA-enabled device count: {cuda_count}")
if cuda_count > 0:
    for i in range(cuda_count):
        device_name = cv.cuda.getDeviceName(i)
        print(f"[OpenCV] Device {i}: {device_name}")
else:
    print("[OpenCV] No CUDA-enabled GPU detected.")

print()

# PyTorch CUDA check
print(f"[PyTorch] CUDA available: {torch.cuda.is_available()}")
print(f"[PyTorch] GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[PyTorch] Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("[PyTorch] No CUDA-enabled GPU detected.")
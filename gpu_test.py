import torch

print("torch version: ", torch.__version__)
print("cuda available: ", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device count: ", torch.cuda.device_count())
    print("device 0: ", torch.cuda.get_device_name(0))
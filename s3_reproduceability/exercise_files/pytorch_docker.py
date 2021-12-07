import torch

if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    print(f"CUDA enabled: {cuda}")
    if cuda:
        print(f"Number of GPUs: {torch.cuda.device_count()}")
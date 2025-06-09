import torch
from colorama import Fore

def cuda_memory_report(detailed=False):
    print(Fore.LIGHTGREEN_EX)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if detailed: print(f'Cuda summary: {torch.cuda.memory_summary()}')
        print(f"Total CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 2} MB")
        print(f"Allocated CUDA memory: {torch.cuda.memory_allocated(0) / 1024 ** 2} MB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated(0) / 1024 ** 2} MB")
    else:
        print("CUDA is not available.")
    print(Fore.RESET)
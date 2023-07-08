import wandb
import torch 
import numpy as np
import random

wb = True

wandb.init(
    project = "SAESUM-Abstractive_Extractive_Summarization",
    config = {
        "epochs":10 
    }
)




def set_seed(seed_value):
    # Set seed for NumPy
    np.random.seed(seed_value)

    # Set seed for Python's random module
    random.seed(seed_value)

    # Set seed for PyTorch
    torch.manual_seed(seed_value)

    # Set seed for GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        # Set the deterministic behavior for cudNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"{seed_value} have been correctly set!")





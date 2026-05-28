import os
from random import random
import torch
import numpy as np
import json

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """Get available device (GPU/CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def load_json_with_encoding(file_path):
    """Load JSON file with proper encoding handling"""
    encodings = ['utf-8-sig', 'utf-8', 'latin1', 'gbk', 'gb2312']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with encoding {encoding}: {e}")
            continue

    raise ValueError(f"Could not decode {file_path} with any of the tried encodings: {encodings}")

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename=None):
    """Save model checkpoint"""
    if filename is None:
        filename = f'checkpoint_epoch_{epoch}.pth'

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return total_params, trainable_params
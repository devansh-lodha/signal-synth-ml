import torch
import torchvision
import torchaudio
import os
from einops import rearrange
from typing import Tuple

def load_image(path: str, device: torch.device) -> torch.Tensor:
    """
    Loads an image from the given path, converts it to a float tensor,
    normalizes to [0, 1], and moves it to the specified device.

    Args:
        path (str): The file path to the image.
        device (torch.device): The device to load the tensor onto.

    Returns:
        torch.Tensor: The loaded image tensor.
    
    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found at: {path}")
    
    img = torchvision.io.read_image(path)
    img_float = img.to(dtype=torch.float32) / 255.0
    return img_float.to(device)

def load_audio(path: str, duration_secs: int = 5) -> Tuple[torch.Tensor, int]:
    """
    Loads an audio file and trims it to a specified duration.

    Args:
        path (str): The file path to the audio file.
        duration_secs (int): The desired duration of the audio in seconds.

    Returns:
        Tuple[torch.Tensor, int]: A tuple containing the audio waveform 
                                     and the sample rate.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found at: {path}")

    waveform, sample_rate = torchaudio.load(path)
    
    # Trim to the desired duration
    num_samples_to_keep = sample_rate * duration_secs
    if waveform.shape[1] > num_samples_to_keep:
        waveform = waveform[:, :num_samples_to_keep]

    # Use only the first channel if stereo
    if waveform.shape[0] > 1:
        waveform = waveform[0, :]
        
    return waveform.squeeze(), sample_rate

def get_image_patch(img: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, int, int]:
    """
    Extracts a random patch of a given size from an image.

    Args:
        img (torch.Tensor): The input image tensor (C, H, W).
        patch_size (int): The height and width of the patch.

    Returns:
        Tuple[torch.Tensor, int, int]: A tuple containing the patch, 
                                       and its starting x and y coordinates.
    """
    _, height, width = img.shape
    max_x = width - patch_size
    max_y = height - patch_size

    start_x = torch.randint(0, max_x + 1, (1,)).item()
    start_y = torch.randint(0, max_y + 1, (1,)).item()
    
    patch = img[:, start_y:start_y + patch_size, start_x:start_x + patch_size]
    return patch, start_x, start_y

def downsample_image(img: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    Downsamples an image by a given scale factor using average pooling.

    Args:
        img (torch.Tensor): The input high-resolution image tensor.
        scale_factor (int): The factor by which to downsample (e.g., 2 for 2x downsampling).

    Returns:
        torch.Tensor: The downsampled low-resolution image tensor.
    """
    new_height = img.shape[1] // scale_factor
    new_width = img.shape[2] // scale_factor
    
    # Add a batch dimension for avg_pool2d
    img_batch = img.unsqueeze(0)
    
    # Use PyTorch's built-in average pooling for efficiency
    downsampled = torch.nn.functional.avg_pool2d(img_batch, kernel_size=scale_factor)
    
    return downsampled.squeeze(0)

def mask_image_by_proportion(img: torch.Tensor, prop: float) -> torch.Tensor:
    """
    Masks an image by replacing a random proportion of pixels with NaN.

    Args:
        img (torch.Tensor): The input image tensor (C, H, W).
        prop (float): The proportion of pixels to mask (e.g., 0.8 for 80%).

    Returns:
        torch.Tensor: The masked image with NaN values.
    """
    img_copy = img.clone()
    # Create a mask for all pixels, then apply it to each channel
    mask = torch.rand(img.shape[1:], device=img.device) < prop
    for c in range(img.shape[0]):
        img_copy[c][mask] = float('nan')
    return img_copy

def mask_image_with_random_block(img: torch.Tensor, block_h: int, block_w: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    Masks an image by placing a NaN block of a given size at a random location.

    Args:
        img (torch.Tensor): The input image tensor (C, H, W).
        block_h (int): The height of the block to remove.
        block_w (int): The width of the block to remove.

    Returns:
        A tuple containing:
        - torch.Tensor: The masked image.
        - Tuple[int, int, int, int]: The coordinates of the block (y_start, y_end, x_start, x_end).
    """
    img_copy = img.clone()
    _, h, w = img.shape
    
    # Choose a random top-left corner for the block
    y_start = torch.randint(0, h - block_h + 1, (1,)).item()
    x_start = torch.randint(0, w - block_w + 1, (1,)).item()
    
    y_end = y_start + block_h
    x_end = x_start + block_w
    
    # Apply NaN mask to the block region in each channel
    img_copy[:, y_start:y_end, x_start:x_end] = float('nan')
    
    return img_copy, (y_start, y_end, x_start, x_end)
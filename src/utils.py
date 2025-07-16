import torch
from sklearn.kernel_approximation import RBFSampler
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.regression import mean_squared_error
from einops import rearrange
from typing import Tuple, List

# --- Coordinate and Pixel Helpers ---
def create_coordinate_map(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Creates a 2D coordinate map for an image, normalized to [-1, 1]."""
    ys = torch.linspace(-1, 1, height, device=device)
    xs = torch.linspace(-1, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([grid_y, grid_x], dim=-1)
    return coords.reshape(-1, 2)

def create_time_coordinates(num_samples: int, device: torch.device) -> torch.Tensor:
    """Creates a 1D tensor of time coordinates, normalized to [-1, 1]."""
    time_steps = torch.linspace(-1, 1, num_samples, device=device)
    return time_steps.unsqueeze(-1)

def get_pixels_from_image(image: torch.Tensor) -> torch.Tensor:
    """Extracts pixel values from an image tensor."""
    return rearrange(image, 'c h w -> (h w) c')


# --- RFF Encoder Class (The Fix) ---
class RFFEncoder:
    """
    A stateful encoder for creating and applying consistent Random Fourier Features.
    This class generates the random projection matrices upon initialization and reuses
    them for every encoding call, ensuring consistency between training and inference.
    """
    def __init__(self, d_in: int, mapping_size_per_scale: int, scales: List[float], device: torch.device):
        """
        Initializes the encoder and creates the fixed random projection matrices.

        Args:
            d_in (int): The input dimension of the coordinates (e.g., 2 for images).
            mapping_size_per_scale (int): Number of Fourier features per scale.
            scales (List[float]): A list of sigma values for the different scales.
            device (torch.device): The device to create tensors on.
        """
        self.d_in = d_in
        self.mapping_size_per_scale = mapping_size_per_scale
        self.scales = scales
        self.device = device
        self.d_out = len(scales) * mapping_size_per_scale * 2
        
        # Create and store the random but fixed projection matrices
        self.B_matrices = []
        for scale in self.scales:
            B = torch.randn((self.d_in, self.mapping_size_per_scale), device=self.device) * scale
            self.B_matrices.append(B)
    
    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encodes coordinates into multi-scale RFFs using the pre-defined matrices.

        Args:
            coords (torch.Tensor): The input coordinates of shape (N, d_in).

        Returns:
            torch.Tensor: The encoded features of shape (N, d_out).
        """
        all_features = []
        for B in self.B_matrices:
            x_proj = (2. * torch.pi * coords) @ B
            features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            all_features.append(features)
        return torch.cat(all_features, dim=-1)


# --- Metrics ---
def calculate_image_metrics(ground_truth: torch.Tensor, prediction: torch.Tensor) -> Tuple[float, float]:
    """Computes RMSE and PSNR for images."""
    prediction = prediction.to(ground_truth.device, dtype=torch.float32)
    gt_flat = ground_truth.reshape(-1)
    pred_flat = prediction.reshape(-1)
    rmse = mean_squared_error(preds=pred_flat, target=gt_flat, squared=False).item()
    psnr = peak_signal_noise_ratio(preds=pred_flat, target=gt_flat, data_range=1.0).item()
    return rmse, psnr

def calculate_audio_metrics(ground_truth: torch.Tensor, prediction: torch.Tensor) -> Tuple[float, float]:
    """Calculates RMSE and SNR for audio."""
    prediction = prediction.to(ground_truth.device, dtype=torch.float32)
    rmse = mean_squared_error(preds=prediction, target=ground_truth, squared=False).item()
    signal_power = torch.mean(ground_truth**2)
    noise_power = torch.mean((ground_truth - prediction)**2)
    snr = 10 * torch.log10(signal_power / noise_power).item() if noise_power > 0 else float('inf')
    return rmse, snr

def extract_known_pixels_and_coords(masked_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the coordinates and pixel values of all non-NaN pixels from a masked image.

    Args:
        masked_img (torch.Tensor): The masked image with NaN values.

    Returns:
        A tuple containing:
        - torch.Tensor: The coordinates of known pixels (N_known, 2).
        - torch.Tensor: The RGB values of known pixels (N_known, 3).
    """
    device = masked_img.device
    h, w = masked_img.shape[1:]
    
    # Check for NaN in the first channel to identify masked pixels
    is_known = ~torch.isnan(masked_img[0])
    
    # Get the (y, x) indices of all known pixels
    known_indices_y, known_indices_x = torch.where(is_known)
    
    # Normalize these integer indices to the [-1, 1] range to create coordinates
    known_coords_y = (known_indices_y / (h - 1)) * 2 - 1
    known_coords_x = (known_indices_x / (w - 1)) * 2 - 1
    known_coords = torch.stack([known_coords_y, known_coords_x], dim=-1)
    
    # Get the pixel values at these known locations
    known_pixels = masked_img[:, known_indices_y, known_indices_x].permute(1, 0)
    
    return known_coords, known_pixels
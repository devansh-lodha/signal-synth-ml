import torch
from tqdm.auto import tqdm
from typing import Tuple, List

def train_rff_model(model: torch.nn.Module, 
                      rff_features: torch.Tensor, 
                      targets: torch.Tensor,
                      epochs: int,
                      learning_rate: float,
                      device: torch.device) -> None:
    """
    Trains a model using Random Fourier Features.

    Args:
        model (torch.nn.Module): The model to train (e.g., LinearRegressionModel).
        rff_features (torch.Tensor): The input features (RFFs).
        targets (torch.Tensor): The target values (e.g., pixel colors, audio amplitudes).
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
        device (torch.device): The device to perform training on.
    """
    model.to(device)
    rff_features = rff_features.to(device)
    targets = targets.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Use tqdm for a professional progress bar
    pbar = tqdm(range(epochs), desc="Training RFF Model")
    for epoch in pbar:
        model.train()
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(rff_features)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update progress bar with the current loss
        pbar.set_postfix(loss=f'{loss.item():.6f}')

def train_matrix_factorization(matrix_channel: torch.Tensor, 
                               rank: int, 
                               epochs: int,
                               learning_rate: float,
                               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs matrix factorization on a single channel of an image using gradient descent.

    Args:
        matrix_channel (torch.Tensor): A 2D tensor representing one image channel (e.g., Red),
                                       which may contain NaN values for missing pixels.
        rank (int): The rank 'r' for the factorization (number of latent features).
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
        device (torch.device): The device to perform training on.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The factorized matrices W and H.
    """
    matrix_channel = matrix_channel.to(device)
    h, w = matrix_channel.shape
    
    # Randomly initialize the factor matrices W and H
    W = torch.randn(h, rank, device=device, requires_grad=True)
    H = torch.randn(rank, w, device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([W, H], lr=learning_rate)
    
    # Create a mask to ignore NaN values in the loss calculation
    mask = ~torch.isnan(matrix_channel)
    
    pbar = tqdm(range(epochs), desc=f"Factorizing Matrix (rank={rank})")
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Reconstruct the matrix
        reconstructed = W @ H
        
        # Calculate loss only on the known pixels
        diff = reconstructed - matrix_channel
        loss = torch.norm(diff[mask])**2 / mask.sum() # MSE on known values
        
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix(loss=f'{loss.item():.6f}')
        
    return W.detach(), H.detach()

def inpaint_mf_svd(masked_channel: torch.Tensor, 
                     rank: int, 
                     iterations: int,
                     device: torch.device) -> torch.Tensor:
    """
    Inpaints a single channel of a masked image using Iterative SVD.
    This is a robust method for matrix completion.

    Args:
        masked_channel (torch.Tensor): A 2D tensor with NaN for missing values.
        rank (int): The rank 'r' for the SVD approximation.
        iterations (int): The number of SVD-update cycles to perform.
        device (torch.device): The device to perform computations on.

    Returns:
        torch.Tensor: The completed/reconstructed image channel.
    """
    masked_channel = masked_channel.to(device)
    
    # Create a mask of the known values
    known_mask = ~torch.isnan(masked_channel)
    
    # 1. Initialize: Fill missing values with the mean of the known values
    mean_val = torch.mean(masked_channel[known_mask])
    completed_channel = torch.where(known_mask, masked_channel, mean_val)
    
    pbar = tqdm(range(iterations), desc=f"Inpainting with SVD (rank={rank})", leave=False)
    for _ in pbar:
        # 2. Approximate: Perform SVD on the currently completed matrix
        U, S, Vh = torch.linalg.svd(completed_channel, full_matrices=False)
        
        # Truncate to the desired rank to get the low-rank approximation
        U_r = U[:, :rank]
        S_r = torch.diag(S[:rank])
        Vh_r = Vh[:rank, :]
        
        low_rank_approx = U_r @ S_r @ Vh_r
        
        # 3. Update: Fill the missing spots in the original matrix with the new values
        # from the low-rank approximation. Keep the original known values.
        completed_channel = torch.where(known_mask, masked_channel, low_rank_approx)

    return completed_channel.detach()

def train_linear_regression_gd(x: torch.Tensor,
                               y: torch.Tensor,
                               algorithm: str,
                               learning_rate: float,
                               epochs: int,
                               batch_size: int = 8,
                               momentum: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[float, float]]]:
    """
    Trains a simple linear regression model (y = t1*x + t0) using various gradient descent algorithms.

    Args:
        x (torch.Tensor): The input data tensor.
        y (torch.Tensor): The target data tensor.
        algorithm (str): The algorithm to use ('batch', 'sgd', 'mini_batch', 'batch_momentum', 'sgd_momentum', 'mini_batch_momentum').
        learning_rate (float): The learning rate.
        epochs (int): The number of epochs to train for.
        batch_size (int): The size of mini-batches.
        momentum (float): The momentum factor (gamma).

    Returns:
        A tuple containing:
        - Final theta0 (intercept).
        - Final theta1 (slope).
        - A history of (theta0, theta1) for each update step.
    """
    device = x.device
    theta0 = torch.tensor([0.0], device=device, requires_grad=True)
    theta1 = torch.tensor([0.0], device=device, requires_grad=True)
    
    # For momentum
    v_theta0 = torch.zeros_like(theta0)
    v_theta1 = torch.zeros_like(theta1)
    
    use_momentum = 'momentum' in algorithm
    
    theta_history = [(theta0.item(), theta1.item())]
    
    pbar = tqdm(range(epochs), desc=f"Training with {algorithm}")
    for epoch in pbar:
        # Create batches
        if 'sgd' in algorithm:
            permutation = torch.randperm(x.size(0))
            batches = permutation.split(1)
        elif 'mini_batch' in algorithm:
            permutation = torch.randperm(x.size(0))
            batches = permutation.split(batch_size)
        else: # batch
            batches = [torch.arange(x.size(0))]
            
        for batch_indices in batches:
            x_batch, y_batch = x[batch_indices], y[batch_indices]

            # Forward pass
            predictions = theta1 * x_batch + theta0
            loss = torch.mean((predictions - y_batch)**2)

            # Backward pass
            loss.backward()

            with torch.no_grad():
                if use_momentum:
                    v_theta0 = momentum * v_theta0 + theta0.grad
                    v_theta1 = momentum * v_theta1 + theta1.grad
                    theta0 -= learning_rate * v_theta0
                    theta1 -= learning_rate * v_theta1
                else:
                    theta0 -= learning_rate * theta0.grad
                    theta1 -= learning_rate * theta1.grad
            
            theta_history.append((theta0.item(), theta1.item()))
            
            # Zero gradients
            theta0.grad.zero_()
            theta1.grad.zero_()

        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return theta0.detach(), theta1.detach(), theta_history

def factorize_patch_gd(patch_channel: torch.Tensor,
                         rank: int,
                         epochs: int,
                         learning_rate: float,
                         device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Factorizes a complete (non-masked) image patch channel using gradient descent.
    This is used for compression, not inpainting.

    Args:
        patch_channel (torch.Tensor): A complete 2D tensor for one channel of the patch.
        rank (int): The rank 'r' for the factorization.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
        device (torch.device): The device to perform computations on.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The factorized matrices W and H.
    """
    patch_channel = patch_channel.to(device)
    h, w = patch_channel.shape
    
    # Randomly initialize the factor matrices
    W = torch.randn(h, rank, device=device, requires_grad=True)
    H = torch.randn(rank, w, device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([W, H], lr=learning_rate)
    criterion = torch.nn.MSELoss()

    pbar = tqdm(range(epochs), desc=f"Factorizing Patch (rank={rank})", leave=False)
    for _ in pbar:
        optimizer.zero_grad()
        
        # Reconstruct the patch channel
        reconstructed = W @ H
        
        # Loss is the Mean Squared Error over the entire patch
        loss = criterion(reconstructed, patch_channel)
        
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix(loss=f'{loss.item():.6f}')
        
    return W.detach(), H.detach()
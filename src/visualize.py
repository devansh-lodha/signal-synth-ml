import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from IPython.display import Audio, display
import imageio
import os
from typing import List, Tuple
from tqdm.auto import tqdm
from matplotlib.patches import Rectangle # <-- THE CRITICAL MISSING IMPORT

def plot_reconstruction(original: torch.Tensor, 
                        reconstructed: torch.Tensor, 
                        task_name: str,
                        save_path: str = None):
    """
    Plots the original and reconstructed images side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(task_name, fontsize=16)
    
    original_np = original.cpu().permute(1, 2, 0).numpy()
    reconstructed_np = reconstructed.cpu().permute(1, 2, 0).numpy()

    axes[0].imshow(np.clip(original_np, 0, 1))
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(np.clip(reconstructed_np, 0, 1))
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        
    plt.show()

def plot_audio(waveform: torch.Tensor, sample_rate: int, title: str):
    """
    Plots an audio waveform and provides an interactive player.
    """
    plt.figure(figsize=(15, 4))
    plt.plot(waveform.cpu().numpy(), alpha=0.8)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    display(Audio(waveform.cpu().numpy(), rate=sample_rate))
    
def generate_gradient_descent_gif(theta_history: List[Tuple[float, float]],
                                  loss_grid: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                  save_path: str):
    """
    Generates and saves a GIF of the gradient descent process on a contour plot.
    """
    INTERCEPT, SLOPE, loss_values = loss_grid
    frames = []
    pbar = tqdm(range(len(theta_history)), desc="Generating GIF frames")
    for i in pbar:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contourf(INTERCEPT, SLOPE, loss_values, levels=50, cmap='viridis', alpha=0.8)
        ax.set_title(f"Gradient Descent Convergence: Iteration {i+1}")
        ax.set_xlabel(r"$\theta_0$ (Intercept)")
        ax.set_ylabel(r"$\theta_1$ (Slope)")
        thetas_np = np.array(theta_history)
        ax.plot(thetas_np[:i+1, 0], thetas_np[:i+1, 1], 'r-o', markersize=4, alpha=0.7)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame_rgba = np.asarray(buf)
        frames.append(frame_rgba[:, :, :3])
        plt.close(fig)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, frames, fps=10, loop=0)
    print(f"GIF saved to {save_path}")

def plot_super_resolution(lr_image: torch.Tensor, 
                          sr_image: torch.Tensor, 
                          hr_image: torch.Tensor,
                          save_path: str = None):
    """
    Plots the low-res input, super-resolved output, and high-res ground truth.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Super-Resolution using SIREN", fontsize=16)
    lr_np = lr_image.cpu().permute(1, 2, 0).numpy()
    sr_np = sr_image.cpu().permute(1, 2, 0).numpy()
    hr_np = hr_image.cpu().permute(1, 2, 0).numpy()
    axes[0].imshow(np.clip(lr_np, 0, 1))
    axes[0].set_title(f"Low-Res Input ({lr_image.shape[2]}x{lr_image.shape[1]})")
    axes[0].axis("off")
    axes[1].imshow(np.clip(sr_np, 0, 1))
    axes[1].set_title(f"Super-Resolved Output ({sr_image.shape[2]}x{sr_image.shape[1]})")
    axes[1].axis("off")
    axes[2].imshow(np.clip(hr_np, 0, 1))
    axes[2].set_title(f"Ground Truth ({hr_image.shape[2]}x{hr_image.shape[1]})")
    axes[2].axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_inpainting_results(original_img: torch.Tensor, 
                              masked_img: torch.Tensor, 
                              recon_img: torch.Tensor, 
                              title: str,
                              save_path: str = None):
    """
    Custom plotting function for inpainting tasks that correctly visualizes NaN values.
    """
    masked_display = masked_img.clone()
    nan_mask = torch.isnan(masked_display)
    masked_display[nan_mask] = 0.5 # Fill with gray
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)
    original_np = original_img.cpu().permute(1, 2, 0).numpy()
    masked_np = masked_display.cpu().permute(1, 2, 0).numpy()
    recon_np = torch.clip(recon_img.cpu().permute(1, 2, 0), 0, 1).numpy()
    axes[0].imshow(original_np)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(masked_np)
    axes[1].set_title("Input with Missing Data")
    axes[1].axis("off")
    axes[2].imshow(recon_np)
    axes[2].set_title("Reconstruction")
    axes[2].axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_inpainting_comparison_with_zoom(original_img, masked_img, recon_mf, recon_siren, block_coords, metrics_mf, metrics_siren):
    """
    Plots a comprehensive comparison for inpainting, including a zoomed-in view of the inpainted region.
    """
    y_start, y_end, x_start, x_end = block_coords
    mf_rmse, mf_psnr = metrics_mf
    siren_rmse, siren_psnr = metrics_siren
    masked_display = masked_img.clone()
    masked_display[:, y_start:y_end, x_start:x_end] = 0.5
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.1])
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(original_img.cpu().permute(1, 2, 0).numpy())
    ax_orig.set_title("Original")
    ax_orig.add_patch(Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, edgecolor='red', facecolor='none', lw=2))
    ax_orig.axis('off')
    ax_masked = fig.add_subplot(gs[0, 1])
    ax_masked.imshow(masked_display.cpu().permute(1, 2, 0).numpy())
    ax_masked.set_title("Input with Missing Block")
    ax_masked.axis('off')
    ax_mf = fig.add_subplot(gs[0, 2])
    ax_mf.imshow(torch.clip(recon_mf.cpu().permute(1, 2, 0), 0, 1).numpy())
    ax_mf.set_title(f"MF (PSNR: {mf_psnr:.2f} dB)")
    ax_mf.add_patch(Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, edgecolor='red', facecolor='none', lw=2))
    ax_mf.axis('off')
    ax_siren = fig.add_subplot(gs[0, 3])
    ax_siren.imshow(torch.clip(recon_siren.cpu().permute(1, 2, 0), 0, 1).numpy())
    ax_siren.set_title(f"SIREN (PSNR: {siren_psnr:.2f} dB)")
    ax_siren.add_patch(Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, edgecolor='red', facecolor='none', lw=2))
    ax_siren.axis('off')
    ax_zoom_orig = fig.add_subplot(gs[1, 0])
    ax_zoom_orig.imshow(original_img[:, y_start:y_end, x_start:x_end].cpu().permute(1, 2, 0).numpy())
    ax_zoom_orig.set_title("Original Patch (Zoomed)")
    ax_zoom_orig.axis('off')
    ax_zoom_mf = fig.add_subplot(gs[1, 2])
    ax_zoom_mf.imshow(torch.clip(recon_mf[:, y_start:y_end, x_start:x_end].cpu().permute(1, 2, 0), 0, 1).numpy())
    ax_zoom_mf.set_title("MF Patch (Zoomed)")
    ax_zoom_mf.axis('off')
    ax_zoom_siren = fig.add_subplot(gs[1, 3])
    ax_zoom_siren.imshow(torch.clip(recon_siren[:, y_start:y_end, x_start:x_end].cpu().permute(1, 2, 0), 0, 1).numpy())
    ax_zoom_siren.set_title("SIREN Patch (Zoomed)")
    ax_zoom_siren.axis('off')
    plt.tight_layout()
    plt.show()

def plot_compression_results(original_img, results_list, patch_coords):
    """
    Creates a comprehensive plot for the image compression task.
    """
    y_start, y_end, x_start, x_end = patch_coords
    num_ranks = len(results_list)
    fig = plt.figure(figsize=(12, 4 + 4 * num_ranks))
    gs = GridSpec(2 + num_ranks, 2, figure=fig, height_ratios=[2] + [2]*num_ranks + [1.5])
    ax_orig = fig.add_subplot(gs[0, :])
    ax_orig.imshow(original_img.cpu().permute(1, 2, 0).numpy())
    ax_orig.set_title("Original Image with Selected 50x50 Patch")
    ax_orig.add_patch(Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, edgecolor='red', facecolor='none', lw=2))
    ax_orig.axis('off')
    for i, result in enumerate(results_list):
        ax_recon = fig.add_subplot(gs[i + 1, :])
        ax_recon.imshow(torch.clip(result['image'].cpu().permute(1, 2, 0), 0, 1).numpy())
        title = f"Rank={result['rank']} | RMSE={result['rmse']:.4f}, PSNR={result['psnr']:.2f} dB"
        ax_recon.set_title(title)
        ax_recon.axis('off')
    ranks = [r['rank'] for r in results_list]
    psnrs = [r['psnr'] for r in results_list]
    rmses = [r['rmse'] for r in results_list]
    ax_rmse = fig.add_subplot(gs[-1, 0])
    ax_psnr = fig.add_subplot(gs[-1, 1])
    ax_rmse.plot(ranks, rmses, 'g-o')
    ax_rmse.set_title('RMSE vs. Rank')
    ax_rmse.set_xlabel('Rank (r)')
    ax_rmse.set_ylabel('RMSE')
    ax_rmse.grid(True, linestyle='--', alpha=0.6)
    ax_psnr.plot(ranks, psnrs, 'b-o')
    ax_psnr.set_title('PSNR vs. Rank')
    ax_psnr.set_xlabel('Rank (r)')
    ax_psnr.set_ylabel('PSNR (dB)')
    ax_psnr.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
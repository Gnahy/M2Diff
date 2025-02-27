import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torch.distributed as dist
from PIL import Image
import random
import cv2
from skimage import measure, morphology
import scipy.io as sio
import logging

def normalize(data, min_val=None, max_val=None):
    if min_val is None:
        min_val = data.min()
    if max_val is None:
        max_val = data.max()
    norm_data = (data - min_val) / (max_val - min_val)
    return norm_data, min_val, max_val

def denormalize(norm_data, min_val, max_val):
    min_val = min_val.view(-1, 1, 1, 1)  # Reshape to [8, 1, 1, 1]
    max_val = max_val.view(-1, 1, 1, 1)  # Reshape to [8, 1, 1, 1]
    return norm_data * (max_val - min_val) + min_val

def set_seed(seed):
    """Set random seed for reproducibility across PyTorch, NumPy, and Python."""
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy seed
    torch.manual_seed(seed)  # PyTorch seed on CPU
    torch.cuda.manual_seed(seed)  # PyTorch seed on GPU (single-GPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch seed on GPU (multi-GPU)
    torch.backends.cudnn.deterministic = True  # Make cudnn deterministic
    torch.backends.cudnn.benchmark = False  # Avoid non-deterministic optimizations

def tensor_to_pil_image(tensor):
    """
    Converts a PyTorch tensor to a PIL Image.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (C, H, W).
        
    Returns:
        PIL.Image: The corresponding PIL Image.
    """
    if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:  # Handle single or 3-channel images
        array = tensor.permute(1, 2, 0).cpu().numpy()  # Change (C, H, W) to (H, W, C)
    elif tensor.ndim == 2:  # Handle grayscale images
        array = tensor.cpu().numpy()
    else:
        raise ValueError("Expected tensor with shape (C, H, W) or (H, W), got shape {}".format(tensor.shape))
    
    # Convert the numpy array to PIL Image
    if array.dtype != np.uint8:  # If dtype is not uint8, scale it to 0-255
        array = (array * 255).astype(np.uint8)
    
    if array.shape[-1] == 1:  # Handle grayscale images
        return Image.fromarray(array.squeeze(), mode='L')
    else:
        return Image.fromarray(array)

def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', rank=rank, world_size=world_size)

def setup_iddpm(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.2:23457', rank=rank, world_size=world_size)

def cleanup():
    # Clean up the process group
    dist.destroy_process_group()

def log_transform(data):
    return np.log1p(data)  # log1p to avoid log(0)

def save_plots(losses, ssims, psnrs, nmses, output_dir):
# def save_plots(losses, output_dir):
    epochs = range(1, len(losses) + 1)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, ssims, label='SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Training SSIM')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, psnrs, label='PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Training PSNR')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, nmses, label='NMSE')
    plt.xlabel('Epoch')
    plt.ylabel('NMSE')
    plt.title('Training NMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()

def save_plots_pet_mri(losses, ssims, psnrs, nmses, output_dir, name):
# def save_plots(losses, output_dir):
    epochs = range(1, len(losses) + 1)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, ssims, label='SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Training SSIM')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, psnrs, label='PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Training PSNR')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, nmses, label='NMSE')
    plt.xlabel('Epoch')
    plt.ylabel('NMSE')
    plt.title('Training NMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, name))
    plt.close()

def save_image_png(lowdose, standarddose, tensor, activation_map, brain_mask, filename):

    # Convert the tensor to a numpy array and remove any singleton dimensions
    image_array = tensor.squeeze().cpu().detach().numpy()
    image_array_ld = lowdose.squeeze().cpu().detach().numpy()
    image_array_sd = standarddose.squeeze().cpu().detach().numpy()

    # Find min and max values for individual scaling of each image
    vmin_ld = image_array_ld.min()
    vmax_ld = image_array_ld.max()

    vmin_sd = image_array_sd.min()
    vmax_sd = image_array_sd.max()

    vmin_gen = image_array.min()
    vmax_gen = image_array.max()

    # Use a common min and max value only for the color bar (optional)
    vmin_global = min(vmin_ld, vmin_sd, vmin_gen)
    vmax_global = max(vmax_ld, vmax_sd, vmax_gen)

    # Determine number of sections (4 or 5 depending on activation_map)
    # if activation_map is not None and brain_mask is not None:
    #     sections = 6
    # elif activation_map is not None:
    #     sections = 5
    # else:
    sections = 4

    plt.figure(figsize=(15, sections))  # Adjust size based on sections

    # Plot the input low-dose image with its own scaling
    plt.subplot(1, sections, 1)
    plt.imshow(image_array_ld, cmap='gray', vmin=vmin_ld, vmax=vmax_ld)
    plt.title('Low Dose Image')
    plt.axis('off')

    # Plot the standard-dose image with its own scaling
    plt.subplot(1, sections, 2)
    img = plt.imshow(image_array_sd, cmap='gray', vmin=vmin_sd, vmax=vmax_sd)
    plt.title('Standard Dose Image')
    plt.axis('off')

    # Plot the generated image with its own scaling
    plt.subplot(1, sections, 3)
    plt.imshow(image_array, cmap='gray', vmin=vmin_gen, vmax=vmax_gen)
    plt.title('Generated Image')
    plt.axis('off')

    # Add a common color bar
    # cbar_ax = plt.subplot(1, sections, 4)
    # plt.colorbar(img, cax=cbar_ax, orientation='vertical')
    # cbar_ax.set_title('Density')

    # if activation_map is not None and brain_mask is not None:
    #     image_array_am = activation_map.squeeze().cpu().detach().numpy()
        
    #     # Plot the activation map
    #     plt.subplot(1, sections, 5)
    #     img = plt.imshow(image_array_am, cmap='BuPu')  # Use 'hot' or another colormap for activation map
    #     plt.title('Activation Map')
    #     plt.axis('off')
        
        # Plot the activation map
        # plt.subplot(1, sections, 6)
        # img = plt.imshow(brain_mask, cmap='gray')  # Use 'hot' or another colormap for activation map
        # plt.title('Brain Mask')
        # plt.axis('off')
    # elif activation_map is not None:
    if activation_map is not None:
        image_array_am = activation_map.squeeze().cpu().detach().numpy()
        
        # Plot the activation map
        plt.subplot(1, sections, 4)
        img = plt.imshow(image_array_am, cmap='BuPu')  # Use 'hot' or another colormap for activation map
        plt.title('Activation Map')
        plt.axis('off')  
    
    if activation_map is None or brain_mask is None:
        sio.savemat(filename + '.mat', {
            'lowdose': image_array_ld,
            'standarddose': image_array_sd,
            'generated': image_array
        })
    else:
        sio.savemat(filename + '.mat', {
            'lowdose': image_array_ld,
            'standarddose': image_array_sd,
            'generated': image_array,
            'mask': brain_mask,
            'error_map': image_array_am
        })

    # Save the image using matplotlib
    plt.savefig(filename+'.png')
    plt.close()  # Close the figure to prevent excessive memory usage

# Brain mask function
def calculate_brain_mask(image, threshold=0.1, min_size=1000):
    """Calculate a brain mask from a brain image using thresholding, morphological operations, 
    and connected component analysis."""
    # Step 1: Normalize the image to [0, 1] if needed
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Step 2: Apply thresholding to separate the brain from the background
    _, binary_mask = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)

    # Step 3: Perform morphological operations to remove small artifacts
    binary_mask = morphology.remove_small_objects(binary_mask.astype(bool), min_size=min_size)
    binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=min_size)
    
    # Step 4: Label connected components and keep the largest component (the brain)
    labels = morphology.label(binary_mask)
    brain_mask = np.zeros_like(binary_mask)
    
    if labels.max() > 0:  # Ensure there are connected components
        # Find the largest connected component (assuming it's the brain)
        largest_component = np.argmax(np.bincount(labels.flat)[1:]) + 1
        brain_mask = (labels == largest_component).astype(np.uint8)
    
    return brain_mask

def get_activation_map(self, layer_name):
    """Register a forward hook to get the activation map from a specific layer."""
    activation = {}

    def hook_fn(module, input, output):
        activation[layer_name] = output.detach()

    layer = dict(self.netG.named_modules())[layer_name]
    layer.register_forward_hook(hook_fn)

    return activation

def save_metrics_to_file(loss_G, ssim, psnr, nmse, lpips, output_dir):
    """Save test metrics to a text file."""
    metrics_path = os.path.join(output_dir, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'Loss_G: {loss_G:.4f}\n')
        f.write(f'SSIM: {ssim:.4f}\n')
        f.write(f'PSNR: {psnr:.4f}\n')
        f.write(f'NMSE: {nmse:.4f}\n')
        f.write(f'LPIPS: {lpips:.4f}\n')
    print(f"Metrics saved to {metrics_path}")

def load_model_weights(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

def save_image(tensor, filename):
    tensor = tensor.squeeze(0).cpu().detach().numpy()
    print(tensor.shape)
    nib.save(nib.Nifti1Image(tensor, np.eye(4)), filename)
    
def combined_loss(outputs, targets, criterion, ssim_metric, psnr_metric):
    mse_loss = criterion(outputs, targets)
    ssim_loss = 1 - ssim_metric(outputs, targets)
    psnr_loss = 1/psnr_metric(outputs, targets)
    combined_loss = 0.5 * mse_loss + 0.3 * ssim_loss + 0.2 * psnr_loss
    return combined_loss

def forward_diffusion(x, t, beta_start=1e-4, beta_end=0.02, lambda_scale=-1):
    # Generate a linearly spaced tensor of betas with length t
    betas = torch.linspace(beta_start, beta_end, t).to(x.device)
    
    # Initialize the noisy image as the original image
    x_noisy = x.clone()
    
    # Compute the signal-dependent scaling factor f(x)
    signal_max = x.max()
    f_x = 1 + lambda_scale * (x / signal_max)
    
    # Apply the diffusion process iteratively
    for step in range(t):
        beta_t = betas[step]
        noise = torch.randn_like(x) * torch.sqrt(beta_t)
        # Apply signal-dependent noise
        if lambda_scale<0:
            x_noisy = x_noisy + noise
        else:
            x_noisy = x_noisy + noise * f_x
    
    return x_noisy, betas

def reverse_diffusion(model, x_noisy, lowdose, scheduler, cond = None, t = 1000, arch = "2D"):
    # t = torch.randint(0,100,(lowdose.shape[0],)).cuda()
    device = lowdose.device
    # Precompute alphas on the appropriate device
    alphas = scheduler.alpha[:t].to(device)
    
    for step in reversed(range(t)):
        time_step = torch.tensor([step], device=device)
        # Generate alpha coefficient for the current step
        if arch == "2D":
            a = alphas[step].view(1, 1, 1, 1).expand(lowdose.shape[0], 1, 1, 1)
        elif arch == "3D":
            a = alphas[step].view(lowdose.shape[0], 1, 1, 1, 1)
        
        # Forward pass of the model
        with torch.no_grad():  # Disable gradient computation for inference
            x_temp = model(x_noisy, time_step, cond)
        
        # Generate random noise and update `x_noisy`
        e = torch.randn_like(lowdose, requires_grad=False)
        x_noisy = (torch.sqrt(a) * x_temp) + (torch.sqrt(1 - a) * e)
    with torch.no_grad():
        x = model(x_noisy, torch.tensor([0], device=device), cond)
    return x

# def reverse_diffusion(model, rank, z, t, scheduler, cond = None, f_x = 1, lambda_scale=0.5):
#     for step in reversed(range(t)):
#         time_step = torch.tensor([step]).to(rank)
#         temp = (scheduler.beta[t]/( (torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t])) ))
#         z = (1/(torch.sqrt(1-scheduler.beta[t])))*z - (temp*model(z.to(rank),time_step, cond))
#         e = torch.randn(z.shape)
#         z = z + (e*f_x*torch.sqrt(scheduler.beta[t]))
#     temp = scheduler.beta[0]/( (torch.sqrt(1-scheduler.alpha[0]))*(torch.sqrt(1-scheduler.beta[0])) )
#     time_step = torch.tensor([0]).to(rank)
#     x = (1/(torch.sqrt(1-scheduler.beta[0])))*z - (temp*model(z.to(rank),time_step, cond))
#     return x

def full_forward_diffusion(standarddose, scheduler, device):
    """
    Full forward diffusion process from t = 1 to T.
    
    Args:
        standarddose (torch.Tensor): Original clean image (batch).
        scheduler (DDPM_Scheduler): Scheduler object to handle alpha values for diffusion.
        device (torch.device): The device to run the process on (GPU/CPU).
    
    Returns:
        torch.Tensor: Noisy image at the last time step.
        list: List of noisy images at each time step.
    """
    T = scheduler.num_time_steps
    x_t = standarddose.to(device)
    noisy_images = []
    
    for step in range(1, T):
        t = torch.tensor([step]).to(device)
        noise = torch.randn_like(x_t).to(device)
        alpha_t = scheduler.alpha[t].view(x_t.shape[0], 1, 1, 1)
        
        # Add noise at step t
        x_t = torch.sqrt(alpha_t) * standarddose + torch.sqrt(1 - alpha_t) * noise
        noisy_images.append(x_t.clone())
    
    return x_t, noisy_images  # Return final noisy image and all intermediate noisy images

def reverse_diffusion_train(model, x_noisy, lowdose, scheduler, cond=None, arch="2D", criterion=None, standarddose=None, optimizer=None):
    T = scheduler.num_time_steps
    total_loss = 0  # To accumulate loss over steps

    for step in reversed(range(1, T)):
        time_step = torch.tensor([step]).cuda()
        x_temp = model(x_noisy, time_step, cond)
        e = torch.randn_like(lowdose, requires_grad=False)
        
        if arch == "2D":
            a = scheduler.alpha[step-1].view(lowdose.shape[0], 1, 1, 1).cuda()
        elif arch == "3D":
            a = scheduler.alpha[step-1].view(lowdose.shape[0], 1, 1, 1, 1).cuda()
        
        x_noisy = (torch.sqrt(a) * x_temp) + (torch.sqrt(1 - a) * e)

        # Compute loss at the current timestep and backpropagate
        if criterion is not None and standarddose is not None:
            loss = criterion(x_temp, standarddose)  # Loss at each timestep
            total_loss += loss.item()  # Accumulate the total loss
            
            # Backpropagation for each step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Clear CUDA memory cache
            torch.cuda.empty_cache()

    # Final step (t=0)
    time_step = torch.tensor([0]).cuda()
    x = model(x_noisy, time_step, cond)

    return x, total_loss  # Return the final reconstructed image and the total loss


def reverse_diffusion_test(model, x_noisy, lowdose, scheduler, cond = None, arch = "2D"):
    T = scheduler.num_time_steps
    for step in reversed(range(1, T)):
        time_step = torch.tensor([step]).cuda()
        x_temp = model(x_noisy, time_step, cond)
        e = torch.randn_like(lowdose, requires_grad=False)
        if (arch == "2D"):
            a = scheduler.alpha[step-1].view(lowdose.shape[0],1,1,1).cuda()
        elif (arch == "3D"):
            a = scheduler.alpha[step-1].view(lowdose.shape[0],1,1,1,1).cuda()
        x_noisy = (torch.sqrt(a)*x_temp) + (torch.sqrt(1-a)*e)
    time_step = torch.tensor([0]).cuda()
    x = model(x_noisy, time_step, cond)
    return x

# def forward_diffusion(x, t, beta_start=1e-4, beta_end=0.02):
#     # Generate a linearly spaced tensor of betas with length t
#     betas = torch.linspace(beta_start, beta_end, t).to(x.device)
    
#     # Initialize the noisy image as the original image
#     x_noisy = x.clone()
    
#     # Apply the diffusion process iteratively
#     for step in range(t):
#         beta_t = betas[step]
#         noise = torch.randn_like(x) * torch.sqrt(beta_t)
#         x_noisy = x_noisy + noise
    
#     return x_noisy, betas

# def reverse_diffusion(model, x_noisy, t, beta, cond = None):
#     for step in reversed(range(t)):
#         time_step = torch.tensor([step], dtype=torch.float32)
#         if cond == None:
#             x_noisy = x_noisy - model(x_noisy, time_step) * beta[step]
#         else:
#             x_noisy = x_noisy - model(x_noisy, time_step, cond) * beta[step]
#     return x_noisy

class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False).cuda()
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False).cuda()
        self.num_time_steps = num_time_steps

    def forward(self, t):
        return self.beta[t], self.alpha[t]

def nmse(pred, target):
    return torch.mean((pred - target) ** 2) / torch.mean(target ** 2)

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    
def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
#%%

def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    state['lossHistory'] = loaded_state['lossHistory']
    return state

def restore_checkpoint_withEval(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    state['lossHistory'] = loaded_state['lossHistory']
    state['evalLossHistory'] = loaded_state['evalLossHistory']
    return state

def restore_checkpoint_BD(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model_br'].load_state_dict(loaded_state['model_br'], strict=False)
    state['model_dr'].load_state_dict(loaded_state['model_dr'], strict=False)
    state['ema_br'].load_state_dict(loaded_state['ema_br'])
    state['ema_dr'].load_state_dict(loaded_state['ema_dr'])
    state['step'] = loaded_state['step']
    state['lossHistory'] = loaded_state['lossHistory']
    return state

def restore_checkpoint_RW(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model_ir'].load_state_dict(loaded_state['model_br'], strict=False)
    state['model_dr'].load_state_dict(loaded_state['model_dr'], strict=False)
    state['ema_ir'].load_state_dict(loaded_state['ema_br'])
    state['ema_dr'].load_state_dict(loaded_state['ema_dr'])
    state['step'] = loaded_state['step']
    state['lossHistory'] = loaded_state['lossHistory']
    state['evalLossHistory'] = loaded_state['evalLossHistory']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step'],
    'lossHistory': state['lossHistory']
  }
  torch.save(saved_state, ckpt_dir)
 

def save_checkpoint_withEval(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step'],
    'lossHistory': state['lossHistory'],
    'evalLossHistory': state['evalLossHistory']
  }
  torch.save(saved_state, ckpt_dir)
    
def save_checkpoint_BD(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model_br': state['model_br'].state_dict(),
    'model_dr': state['model_dr'].state_dict(),
    'ema_br': state['ema_br'].state_dict(),
    'ema_dr': state['ema_dr'].state_dict(),
    'step': state['step'],
    'lossHistory': state['lossHistory']  
  }
  torch.save(saved_state, ckpt_dir)
    
def save_checkpoint_RW(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model_br': state['model_ir'].state_dict(),
    'model_dr': state['model_dr'].state_dict(),
    'ema_br': state['ema_ir'].state_dict(),
    'ema_dr': state['ema_dr'].state_dict(),
    'step': state['step'],
    'lossHistory': state['lossHistory'],
    'evalLossHistory': state['evalLossHistory']
  }
  torch.save(saved_state, ckpt_dir)
  
class ExponentialMovingAverage:
  """
  Maintains (exponential) moving average of a set of parameters.
  """

  def __init__(self, parameters, decay, use_num_updates=True):
    """
    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the result of
        `model.parameters()`.
      decay: The exponential decay.
      use_num_updates: Whether to use number of updates when computing
        averages.
    """
    if decay < 0.0 or decay > 1.0:
      raise ValueError('Decay must be between 0 and 1')
    self.decay = decay
    self.num_updates = 0 if use_num_updates else None
    self.shadow_params = [p.clone().detach()
                          for p in parameters if p.requires_grad]
    self.collected_params = []

  def update(self, parameters):
    """
    Update currently maintained parameters.

    Call this every time the parameters are updated, such as the result of
    the `optimizer.step()` call.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the same set of
        parameters used to initialize this object.
    """
    decay = self.decay
    if self.num_updates is not None:
      self.num_updates += 1
      decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
    one_minus_decay = 1.0 - decay
    with torch.no_grad():
      parameters = [p for p in parameters if p.requires_grad]
      for s_param, param in zip(self.shadow_params, parameters):
        s_param.sub_(one_minus_decay * (s_param - param))

  def copy_to(self, parameters):
    """
    Copy current parameters into given collection of parameters.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored moving averages.
    """
    parameters = [p for p in parameters if p.requires_grad]
    for s_param, param in zip(self.shadow_params, parameters):
      if param.requires_grad:
        param.data.copy_(s_param.data)

  def store(self, parameters):
    """
    Save the current parameters for restoring later.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        temporarily stored.
    """
    self.collected_params = [param.clone() for param in parameters]

  def restore(self, parameters):
    """
    Restore the parameters stored with the `store` method.
    Useful to validate the model with EMA parameters without affecting the
    original optimization process. Store the parameters before the
    `copy_to` method. After validation (or model saving), use this to
    restore the former parameters.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored parameters.
    """
    for c_param, param in zip(self.collected_params, parameters):
      param.data.copy_(c_param.data)

  def state_dict(self):
    return dict(decay=self.decay, num_updates=self.num_updates,
                shadow_params=self.shadow_params)

  def load_state_dict(self, state_dict):
    self.decay = state_dict['decay']
    self.num_updates = state_dict['num_updates']
    self.shadow_params = state_dict['shadow_params']
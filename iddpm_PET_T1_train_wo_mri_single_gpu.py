#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:47:38 2024

@author: gyar0001
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:40:12 2024

@author: gyar0001
"""

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch.optim as optim
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import torch.multiprocessing as mp
from helping.utils import setup_iddpm, set_seed, cleanup, mkdir, save_model, nmse, save_plots_pet_mri, load_model_weights, save_image, save_image_png, DDPM_Scheduler, tensor_to_pil_image, calculate_brain_mask
from helping.dataloaders import BiTaskPETDatasetMat
from models.denoising_diffusion_pytorch import Unet, GaussianDiffusion 
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
import torchvision.transforms as T
import argparse
import lpips
import numpy as np
import sys
import csv
from improved_diffusion_PET_T1_train_wo_mri.script_util import create_bi_task_model_and_diffusion, bi_task_model_and_diffusion_defaults
import time

# Add argparse for parsing command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPM model with custom hyperparameters")

    # Paths
    parser.add_argument('--experiment_name', type=str, default='test', help='Directory to save output images and plots for testing')
    parser.add_argument('--train_data_path', type=str, default='../../../../../data/gyar0001/dataset_discrete/mat/train')
    parser.add_argument('--val_data_path', type=str, default='../../../../../data/gyar0001/dataset_discrete/mat/val')
    parser.add_argument('--test_data_path', type=str, default='../../../../../data/gyar0001/dataset_discrete/mat/test')
    parser.add_argument('--checkpoint_path', type=str, default='../../../../../data/gyar0001/iddpm_dir_PET_T1_train_wo_mri/weights')
    parser.add_argument('--train_output_dir', type=str, default='../../../../../data/gyar0001/iddpm_dir_PET_T1_train_wo_mri/results_train')
    parser.add_argument('--test_output_dir', type=str, default='../../../../../data/gyar0001/iddpm_dir_PET_T1_train_wo_mri/results_test')
    parser.add_argument('--pretrained_model_path', type=str, default='../../../../../data/gyar0001/iddpm_dir_PET_T1_train_wo_mri')
    
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for the model to prevent overfitting')

    # Diffusion model hyperparameters
    parser.add_argument('--beta_schedule', type=str, default="linear", choices=["linear", "cosine"], help="Beta schedule type")
    parser.add_argument('--timesteps', type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument('--model_mean_type', type=str, default="epsilon", choices=["epsilon", "start_x", "previous_x"], help="Model output type")
    parser.add_argument('--model_var_type', type=str, default="fixed_small", choices=["fixed_small", "fixed_large", "learned", "learned_range"], help="Model variance type")
    parser.add_argument('--loss_type', type=str, default="mse", choices=["mse", "rescaled_mse", "kl", "rescaled_kl"], help="Loss function type")
    parser.add_argument('--clip_denoised', type=bool, default=True, help="Clip output")
    
    return parser.parse_args()

def list_all_layers(model):
    for name, layer in model.named_modules():
        print(f"{name}: {layer.__class__.__name__}")

def validate(model, val_dataloader, diffusion, scheduler, criterion, ssim_metric, psnr_metric, output_dir, num_time_steps=1000):
    """
    Validate the bi-task model on the validation dataset, compute losses and metrics, and save predicted images.

    Args:
        rank (int): GPU rank for distributed training.
        model (torch.nn.Module): The trained model.
        val_dataloader (torch.utils.data.DataLoader): Dataloader for the validation set.
        scheduler (DDPM_Scheduler): Scheduler for diffusion process.
        criterion (torch.nn.Module): Loss function.
        ssim_metric (torchmetrics.Metric): SSIM metric for validation.
        psnr_metric (torchmetrics.Metric): PSNR metric for validation.
        output_dir (str): Directory to save predicted images.
        num_time_steps (int): Number of time steps in the diffusion process.

    Returns:
        dict: Dictionary containing average validation loss and metrics (SSIM, PSNR, NMSE) for both PET and MRI pathways.
    """
    model.eval()  # Set model to evaluation mode
    running_val_bias_loss = 0.0
    running_val_loss_pet = 0.0
    running_val_loss_mri = 0.0
    running_ssim_pet = 0.0
    running_ssim_mri = 0.0
    running_psnr_pet = 0.0
    running_psnr_mri = 0.0
    running_nmse_pet = 0.0
    running_nmse_mri = 0.0

    # Ensure the output directory exists for saving images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    count = 0
    with torch.no_grad():  # Disable gradient calculation
        for i, (lowdose, standarddose, mri) in enumerate(val_dataloader):
            if (i + 1) % 10 == 0:
                lowdose, standarddose, mri = lowdose.cuda(), standarddose.cuda(), mri.cuda()
                
                if random.random() < 0.4:
                    mri_flag = 0   # simulate missing MRI
                else:
                    mri_flag = 1
                
                # Sample from the diffusion model conditioned on lowdose and mri
                output = diffusion.p_sample_loop(
                    model,
                    standarddose.shape,
                    model_kwargs={"lowdose": lowdose, "mri": mri, "mri_flag": mri_flag}
                )
                pet_pred = output['pet']
                mri_pred = output['mri']
                
                # Compute validation loss for PET and MRI pathways
                val_bias_loss = criterion(pet_pred, mri_pred)
                running_val_bias_loss += val_bias_loss.item() 
                
                val_loss_pet = criterion(pet_pred, standarddose)
                val_loss_mri = criterion(mri_pred, standarddose)
                running_val_loss_pet += val_loss_pet.item()
                running_val_loss_mri += val_loss_mri.item()
    
                # Compute metrics for PET pathway
                running_ssim_pet += ssim_metric(pet_pred, standarddose).item()
                running_psnr_pet += psnr_metric(pet_pred, standarddose).item()
                running_nmse_pet += nmse(pet_pred, standarddose).item()
                
                # Compute metrics for MRI pathway
                running_ssim_mri += ssim_metric(mri_pred, standarddose).item()
                running_psnr_mri += psnr_metric(mri_pred, standarddose).item()
                running_nmse_mri += nmse(mri_pred, standarddose).item()
                
                # Save images for visualization
                error_map_pet = torch.abs(standarddose - pet_pred)
                error_map_mri = torch.abs(standarddose - mri_pred)
                for idx in range(len(pet_pred[:, 0, 0])):
                    # Save PET results
                    pet_output_filename = os.path.join(output_dir, f'pet_output_batch_{idx}_image_{count + 1}')
                    save_image_png(
                        lowdose[idx], standarddose[idx], pet_pred[idx], error_map_pet[idx], 
                        None, pet_output_filename + '.png'
                    )
                    print(f'PET image saved to {pet_output_filename}')
                    
                    # Save MRI results
                    mri_output_filename = os.path.join(output_dir, f'mri_output_batch_{idx}_image_{count + 1}')
                    save_image_png(
                        lowdose[idx], standarddose[idx], mri_pred[idx], error_map_mri[idx], 
                        None, mri_output_filename + '.png'
                    )
                    print(f'MRI image saved to {mri_output_filename}')
                count += 1

    # Compute averages over the validation set for PET and MRI pathways
    avg_val_bias_loss = running_val_bias_loss / count
    avg_val_loss_pet = running_val_loss_pet / count
    avg_val_loss_mri = running_val_loss_mri / count
    avg_ssim_pet = running_ssim_pet / count
    avg_ssim_mri = running_ssim_mri / count
    avg_psnr_pet = running_psnr_pet / count
    avg_psnr_mri = running_psnr_mri / count
    avg_nmse_pet = running_nmse_pet / count
    avg_nmse_mri = running_nmse_mri / count

    return {
        'val_bias_loss': avg_val_bias_loss,
        'val_loss_pet': avg_val_loss_pet,
        'val_loss_mri': avg_val_loss_mri,
        'ssim_pet': avg_ssim_pet,
        'ssim_mri': avg_ssim_mri,
        'psnr_pet': avg_psnr_pet,
        'psnr_mri': avg_psnr_mri,
        'nmse_pet': avg_nmse_pet,
        'nmse_mri': avg_nmse_mri,
    }

import random

def train(model, diffusion, train_dataloader, val_dataloader, test_loader, args):

    # Create model and move it to the GPU corresponding to `rank`
    scheduler = DDPM_Scheduler(num_time_steps=args.timesteps)
    ddp_model = model.cuda()
    # ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.learning_rate)
    # if args.lr_policy == "cosine":
    #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
    # elif args.lr_policy == "linear":
    #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
    criterion = nn.MSELoss()  # Loss will now compare predicted image to actual image
    ssim_metric = StructuralSimilarityIndexMeasure().cuda()
    psnr_metric = PeakSignalNoiseRatio().cuda()
    
    best_val_loss = float('inf')
    early_stopping_counter = 0

    # Initialize lists to store validation metrics
    val_losses_pet, val_ssims_pet, val_psnrs_pet, val_nmses_pet, val_losses_mri, val_ssims_mri, val_psnrs_mri, val_nmses_mri, val_bias_losses = [], [], [], [], [], [], [], [], []
    print("Training Started!!!")
    
    with open(f'{args.checkpoint_path}/avg_train_loss.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Avg Train Loss'])
    
    # Training loop
    for epoch in range(args.num_epochs):
        running_train_loss = 0
        model.train()
        epoch_start_time = time.time()
        for lowdose, standarddose, mri in train_dataloader:
            iter_start_time = time.time()
            lowdose, standarddose, mri = lowdose.cuda(), standarddose.cuda(), mri.cuda()
            optimizer.zero_grad()

            # Forward pass using the diffusion's training loss method
            t = torch.randint(0, diffusion.num_timesteps, (lowdose.shape[0],)).cuda()
            x_t = diffusion.q_sample(x_start=standarddose, t=t)
            if random.random() < 0.4:
                mri_flag = 0   # simulate missing MRI
            else:
                mri_flag = 1
            loss_dict = diffusion.training_losses(model=ddp_model, x_start=standarddose, t=t, model_kwargs={"lowdose": lowdose, "mri": mri, "mri_flag": mri_flag})
            # Extract losses
            pet_loss = loss_dict["pet_loss"].mean()
            mri_loss = loss_dict["mri_loss"].mean()
            
            # Generate p_sample predictions
            predicted_output = diffusion.p_sample(
                model=ddp_model, x=x_t, t=t, clip_denoised=True, model_kwargs={"lowdose": lowdose, "mri": mri, "mri_flag": mri_flag}
            )
            predicted_pet_x_t_minus_1 = predicted_output['pet']["sample"]
            predicted_mri_x_t_minus_1 = predicted_output['mri']["sample"]
            
            bias_loss = criterion(predicted_pet_x_t_minus_1, predicted_mri_x_t_minus_1)
            
            # Combine losses with weights
            loss = 0.4 * pet_loss + 0.4 * mri_loss + 0.2 * bias_loss  # Adjust weights as needed
            
            running_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            iter_time = time.time() - iter_start_time
            # print(f"Iteration time: {iter_time:.4f}s")
        
        avg_train_loss = running_train_loss/len(train_dataloader)
        print(f'average taining loss = {avg_train_loss}')
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch + 1}/{args.num_epochs}] complete. Time: {epoch_time:.4f}s")

        with open(f'{args.checkpoint_path}/avg_train_loss.csv', mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss])
        
        if epoch >= 10:
            # Define where to save validation images
            print(f'Validation running for epoch = {epoch}, as average train loss = {avg_train_loss}')
            val_output_dir = os.path.join(args.train_output_dir, f'epoch_{epoch + 1}')
            
            # Perform validation and save predicted images
            val_metrics = validate(model, val_dataloader, diffusion, scheduler, criterion, ssim_metric, psnr_metric, val_output_dir, args.timesteps)
            
            # Track validation metrics
            val_bias_losses.append(val_metrics['val_bias_loss'])
            val_losses_pet.append(val_metrics['val_loss_pet'])
            val_ssims_pet.append(val_metrics['ssim_pet'])
            val_psnrs_pet.append(val_metrics['psnr_pet'])
            val_nmses_pet.append(val_metrics['nmse_pet'])
            val_losses_mri.append(val_metrics['val_loss_mri'])
            val_ssims_mri.append(val_metrics['ssim_mri'])
            val_psnrs_mri.append(val_metrics['psnr_mri'])
            val_nmses_mri.append(val_metrics['nmse_mri'])
            
            print(f'Epoch [{epoch + 1}/{args.num_epochs}], Validation Bias Loss: {val_metrics["val_bias_loss"]}, Validation Loss PET: {val_metrics["val_loss_pet"]:.4f}, SSIM PET: {val_metrics["ssim_pet"]:.4f}, PSNR PET: {val_metrics["psnr_pet"]:.4f}, NMSE PET: {val_metrics["nmse_pet"]:.4f},\
                  , Validation Loss MRI: {val_metrics["val_loss_mri"]:.4f}, SSIM MRI: {val_metrics["ssim_mri"]:.4f}, PSNR MRI: {val_metrics["psnr_mri"]:.4f}, NMSE MRI: {val_metrics["nmse_mri"]:.4f}')
            
            # Early stopping logic based on validation loss
            validation_loss = 0.4 * val_metrics['val_loss_pet'] + 0.4 * val_metrics['val_loss_mri'] + 0.2 * val_metrics['val_bias_loss']
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                early_stopping_counter = 0
                save_model(model, args.checkpoint_path+'/iddpm_weights.pth')  # Save the best model
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        # lr_scheduler.step()

    # Plot and save validation metrics
    save_plots_pet_mri(val_losses_pet, val_ssims_pet, val_psnrs_pet, val_nmses_pet, args.train_output_dir, "training_metices_pet.png")
    save_plots_pet_mri(val_losses_pet, val_ssims_pet, val_psnrs_pet, val_nmses_pet, args.train_output_dir, "training_metices_mri.png")

    # cleanup()

    return val_losses_pet, val_ssims_pet, val_psnrs_pet, val_nmses_pet

def combined_loss(outputs, targets, criterion, ssim_metric, psnr_metric):
    mse_loss = criterion(outputs, targets)
    ssim_loss = 1 - ssim_metric(outputs, targets)
    psnr_loss = 1/psnr_metric(outputs, targets)
    combined_loss = 0.5 * mse_loss + 0.3 * ssim_loss + 0.2 * psnr_loss
    return combined_loss

def test(model, diffusion, test_loader, mri_flag, args):
    """
    Test the bi-task diffusion model and compute metrics for PET and MRI outputs.

    Args:
        model (torch.nn.Module): The trained model.
        diffusion (GaussianDiffusion): Diffusion process for generating outputs.
        test_loader (torch.utils.data.DataLoader): Dataloader for the test set.
        args: Command-line arguments.

    Returns:
        None
    """
    # Load the model weights
    load_model_weights(model, args.checkpoint_path + '/iddpm_weights.pth')
    
    # Create the test output directory if it doesn't exist
    if not os.path.exists(args.test_output_dir):
        os.makedirs(args.test_output_dir)
    
    # Initialize metrics
    ssim_metric = StructuralSimilarityIndexMeasure().cuda()
    psnr_metric = PeakSignalNoiseRatio().cuda()
    lpips_metric = lpips.LPIPS(net='alex').cuda()
    
    # Separate totals for PET and MRI
    total_metrics = {
        "ssim_pet": 0.0,
        "ssim_mri": 0.0,
        "psnr_pet": 0.0,
        "psnr_mri": 0.0,
        "nmse_pet": 0.0,
        "nmse_mri": 0.0,
        "lpips_pet": 0.0,
        "lpips_mri": 0.0
    }
    overall_count = 0

    # Loop through the test data
    for idx, (lowdose, standarddose, mri) in enumerate(test_loader):
        lowdose, standarddose, mri = lowdose.cuda(), standarddose.cuda(), mri.cuda()
        
        # Sample from the diffusion model using lowdose and mri as conditions
        pred = diffusion.p_sample_loop(
            model,
            standarddose.shape,
            model_kwargs={"lowdose": lowdose, "mri": mri, "mri_flag": mri_flag},
            clip_denoised=args.clip_denoised
        )
        pet_pred = pred['pet']
        mri_pred = pred['mri']
        
        batch_count = 0
        for i in range(len(pet_pred)):
            sd_np = standarddose[i, :, :].cpu().numpy()  # Standard-dose (ground truth)

            # Calculate brain mask for the ground truth
            brain_mask = calculate_brain_mask(sd_np)

            # Skip the image if the brain mask is empty
            if np.sum(brain_mask) == 0:
                print(f"Skipping image {i + 1} in batch {idx + 1} due to empty brain mask.")
                continue

            # Apply brain mask to both real_B and fake_B
            masked_sd = standarddose[i, :, :] * torch.tensor(brain_mask, dtype=torch.float32).cuda()
            masked_pet_pred = pet_pred[i, :, :] * torch.tensor(brain_mask, dtype=torch.float32).cuda()
            masked_mri_pred = mri_pred[i, :, :] * torch.tensor(brain_mask, dtype=torch.float32).cuda()
            
            # Compute metrics for PET output
            total_metrics["ssim_pet"] += ssim_metric(masked_pet_pred.unsqueeze(0), masked_sd.unsqueeze(0)).item()
            total_metrics["psnr_pet"] += psnr_metric(masked_pet_pred.unsqueeze(0), masked_sd.unsqueeze(0)).item()
            total_metrics["nmse_pet"] += nmse(masked_pet_pred, masked_sd).item()
            total_metrics["lpips_pet"] += lpips_metric(masked_pet_pred.unsqueeze(0), masked_sd.unsqueeze(0)).mean().item()

            # Compute metrics for MRI output
            total_metrics["ssim_mri"] += ssim_metric(masked_mri_pred.unsqueeze(0), masked_sd.unsqueeze(0)).item()
            total_metrics["psnr_mri"] += psnr_metric(masked_mri_pred.unsqueeze(0), masked_sd.unsqueeze(0)).item()
            total_metrics["nmse_mri"] += nmse(masked_mri_pred, masked_sd).item()
            total_metrics["lpips_mri"] += lpips_metric(masked_mri_pred.unsqueeze(0), masked_sd.unsqueeze(0)).mean().item()

            # Calculate the error map
            error_map_pet = torch.abs(masked_sd - masked_pet_pred)
            error_map_mri = torch.abs(masked_sd - masked_mri_pred)

            print(f'Test Image {idx + 1} for PET - SSIM: {ssim_metric(masked_pet_pred.unsqueeze(0), masked_sd.unsqueeze(0)).item():.4f},\
                  PSNR: {psnr_metric(masked_pet_pred.unsqueeze(0), masked_sd.unsqueeze(0)).item():.4f},\
                      NMSE: {nmse(masked_pet_pred, masked_sd).item():.4f},\
                          LPIPS: {lpips_metric(masked_pet_pred.unsqueeze(0), masked_sd.unsqueeze(0)).mean().item():.4f}')
            print(f'Test Image {idx + 1} for MRI - SSIM: {ssim_metric(masked_mri_pred.unsqueeze(0), masked_sd.unsqueeze(0)).item():.4f},\
                  PSNR: {psnr_metric(masked_mri_pred.unsqueeze(0), masked_sd.unsqueeze(0)).item():.4f},\
                      NMSE: {nmse(masked_mri_pred, masked_sd).item():.4f},\
                          LPIPS: {lpips_metric(masked_mri_pred.unsqueeze(0), masked_sd.unsqueeze(0)).mean().item():.4f}')

            # Save images for PET
            pet_output_filename = os.path.join(args.test_output_dir+str(mri_flag), f'pet_test_batch_{idx}_image_{i}')
            save_image_png(lowdose[i, :, :], masked_sd, masked_pet_pred, error_map_pet, brain_mask[0, :, :], pet_output_filename + '.png')
            print(f'PET output saved to {pet_output_filename}.png')

            # Save images for MRI
            mri_output_filename = os.path.join(args.test_output_dir+str(mri_flag), f'mri_test_batch_{idx}_image_{i}')
            save_image_png(lowdose[i, :, :], masked_sd, masked_mri_pred, error_map_mri, brain_mask[0, :, :], mri_output_filename + '.png')
            print(f'MRI output saved to {mri_output_filename}.png')

            overall_count += 1
            batch_count += 1

    # Compute averages for PET and MRI
    avg_metrics = {key: total / overall_count for key, total in total_metrics.items()}

    # Print results
    print(f'Test Results for PET - SSIM: {avg_metrics["ssim_pet"]:.4f}, PSNR: {avg_metrics["psnr_pet"]:.4f}, NMSE: {avg_metrics["nmse_pet"]:.4f}, LPIPS: {avg_metrics["lpips_pet"]:.4f}')
    print(f'Test Results for MRI - SSIM: {avg_metrics["ssim_mri"]:.4f}, PSNR: {avg_metrics["psnr_mri"]:.4f}, NMSE: {avg_metrics["nmse_mri"]:.4f}, LPIPS: {avg_metrics["lpips_mri"]:.4f}')

def main():
    
    # Set the seed for reproducibility
    seed = 42  # You can also get this value from the command line arguments if needed
    set_seed(seed)
    
    args = parse_args()
    
    mkdir(args.train_output_dir)
    mkdir(args.test_output_dir)
    mkdir(args.checkpoint_path)
    
    gpus = list(map(int, args.gpu_ids.split(',')))  # For example, use GPU 0 and GPU 2
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" #",".join(map(str, gpus))

    # Assuming you have a training dataset
    
    lowdose_path = os.path.join(args.train_data_path, 'LD')
    standarddose_path = os.path.join(args.train_data_path, 'SD')
    mri_path = os.path.join(args.train_data_path, 'T1')
    train_dataset = BiTaskPETDatasetMat(lowdose_path, standarddose_path, mri_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Assuming you have a training dataset
    
    lowdose_path = os.path.join(args.val_data_path, 'LD')
    standarddose_path = os.path.join(args.val_data_path, 'SD')
    mri_path = os.path.join(args.val_data_path, 'T1')
    val_dataset = BiTaskPETDatasetMat(lowdose_path, standarddose_path, mri_path)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
    
    lowdose_path = os.path.join(args.test_data_path, 'LD')
    standarddose_path = os.path.join(args.test_data_path, 'SD')
    mri_path = os.path.join(args.test_data_path, 'T1')
    test_dataset = BiTaskPETDatasetMat(lowdose_path, standarddose_path, mri_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model_diffusion_params = bi_task_model_and_diffusion_defaults()
    print(model_diffusion_params)
    model_diffusion_params.update({
        "image_size": 256,  # Custom image size
        "noise_schedule": args.beta_schedule,
        "dropout": args.dropout
    })
    
    # Create bi-task UNet model and diffusion
    model, diffusion = create_bi_task_model_and_diffusion(**model_diffusion_params)
    
    model_path = args.pretrained_model_path + '/iddpm_weights.pth'
    print(model_path)
    if model_path and os.path.isfile(model_path):
        print(f"Loading pre-trained weights from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
    
        # If the checkpoint has a state_dict key (common in torch.save({...})), handle it
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Assume checkpoint itself is the state_dict
            model.load_state_dict(checkpoint)
    
        print("Pre-trained model loaded successfully")
    else:
        print("No pre-trained model path provided, training from scratch")
    
    mri_flag_list = [1]
    for i in mri_flag_list:
    
        log_file_path = os.path.join(args.checkpoint_path, f'logs_{args.experiment_name}_mri_flag_{i}.txt')
    
        # Open the file and redirect stdout
        with open(log_file_path, 'w') as f:
            sys.stdout = f  # Redirect all print statements to this file
            try:
                model = model.cuda()
                test(model, diffusion, test_loader, i, args)
            finally:
                sys.stdout = sys.__stdout__
    
if __name__ == "__main__":
    main()



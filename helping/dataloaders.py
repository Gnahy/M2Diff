import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
import scipy
from helping.utils import normalize
from PIL import Image
import torch.nn.functional as F

class PETDatasetNifti(Dataset):
    def __init__(self, lowdose_dir, standarddose_dir, transform=None):
        self.lowdose_dir = Path(lowdose_dir)
        self.standarddose_dir = Path(standarddose_dir)
        self.transform = transform
        self.lowdose_files = sorted(self.lowdose_dir.glob('*.nii'))
        self.standarddose_files = sorted(self.standarddose_dir.glob('*.nii'))

    def __len__(self):
        return len(self.lowdose_files)
    
    def __getitem__(self, idx):
        lowdose_path = self.lowdose_files[idx]
        standarddose_path = self.standarddose_files[idx]
        
        low_dose_data = nib.load(lowdose_path).get_fdata()#[:, :, 16:80]
        standard_dose_data = nib.load(standarddose_path).get_fdata()#[:, :, 16:80]
        
        # Apply log transform
        # low_dose_data = log_transform(low_dose_data)
        # standard_dose_data = log_transform(standard_dose_data)
        
        # Normalize data
        low_dose_data, low_dose_min, low_dose_max = normalize(low_dose_data)
        standard_dose_data, standard_dose_min, standard_dose_max = normalize(standard_dose_data)
        
        lowdose_tensor = torch.tensor(low_dose_data, dtype=torch.float32).unsqueeze(0)
        standarddose_tensor = torch.tensor(standard_dose_data, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            lowdose_tensor = self.transform(lowdose_tensor)
            standarddose_tensor = self.transform(standarddose_tensor)
        
        return lowdose_tensor, standarddose_tensor

class PETDatasetMat(Dataset):
    def __init__(self, low_dose_dir, standard_dose_dir, im_size = 256, transform=None):
        all_low_dose_files = sorted([os.path.join(low_dose_dir, f) for f in os.listdir(low_dose_dir) if f.endswith('.mat')])
        all_standard_dose_files = sorted([os.path.join(standard_dose_dir, f) for f in os.listdir(standard_dose_dir) if f.endswith('.mat')])

        self.low_dose_files = []
        self.standard_dose_files = []
        # Pre-filter images to remove empty ones
        for ld_file, sd_file in zip(all_low_dose_files, all_standard_dose_files):
            ld_data = scipy.io.loadmat(ld_file)['slice_data']
            sd_data = scipy.io.loadmat(sd_file)['slice_data']
            if not (np.all(ld_data == 0) or np.all(sd_data == 0)):  # Only keep non-empty images
                self.low_dose_files.append(ld_file)
                self.standard_dose_files.append(sd_file)
            else:
                print(f"Skipping empty image pair: {ld_file}, {sd_file}")
        self.transform = transform
        self.im_size = im_size

    def __len__(self):
        return len(self.low_dose_files)

    def __getitem__(self, idx):
        low_dose_data = scipy.io.loadmat(self.low_dose_files[idx])['slice_data'] #[:80, :96]
        standard_dose_data = scipy.io.loadmat(self.standard_dose_files[idx])['slice_data'] #[:80, :96]

        # Check if images are empty (all zero values)
        # if np.all(low_dose_data == 0) or np.all(standard_dose_data == 0):
        #     print(f"Skipping empty image at index {idx}: {self.low_dose_files[idx]}")
        #     return self.__getitem__((idx + 1) % len(self))  # Get the next valid sample

        low_dose_data = np.expand_dims(low_dose_data, axis=0)  # Convert to 4D tensor
        standard_dose_data = np.expand_dims(standard_dose_data, axis=0)  # Convert to 4D tensor

        # Normalize data
        low_dose_data, low_dose_min, low_dose_max = normalize(low_dose_data)
        standard_dose_data, standard_dose_min, standard_dose_max = normalize(standard_dose_data)

        # Get the original height and width
        h, w = low_dose_data.shape[-2], low_dose_data.shape[-1]

        #Calculate the padding needed to reach 256x256
        pad_h = (self.im_size - h) // 2
        pad_w = (self.im_size - w) // 2
        pad_h_extra = (self.im_size - h) % 2
        pad_w_extra = (self.im_size - w) % 2

        low_dose_data = np.pad(low_dose_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)
        standard_dose_data = np.pad(standard_dose_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)

        low_dose_data = torch.tensor(low_dose_data, dtype=torch.float32)
        standard_dose_data = torch.tensor(standard_dose_data, dtype=torch.float32)

        if self.transform:
            low_dose_data = self.transform(low_dose_data)
            standard_dose_data = self.transform(standard_dose_data)
        
        return low_dose_data, standard_dose_data
    
class BiTaskPETDatasetMat(Dataset):
    def __init__(self, low_dose_dir, standard_dose_dir, mri_dir, transform=None):
        self.low_dose_files = sorted([os.path.join(low_dose_dir, f) for f in os.listdir(low_dose_dir) if f.endswith('.mat')])
        self.standard_dose_files = sorted([os.path.join(standard_dose_dir, f) for f in os.listdir(standard_dose_dir) if f.endswith('.mat')])
        self.mri_files = sorted([os.path.join(mri_dir, f) for f in os.listdir(mri_dir) if f.endswith('.mat')])
        self.transform = transform

    def __len__(self):
        return len(self.low_dose_files)

    def __getitem__(self, idx):
        low_dose_data = scipy.io.loadmat(self.low_dose_files[idx])['slice_data'] #[:80, :96]
        standard_dose_data = scipy.io.loadmat(self.standard_dose_files[idx])['slice_data'] #[:80, :96]
        mri_data = scipy.io.loadmat(self.mri_files[idx])['slice_data'] #[:80, :96]

        low_dose_data = np.expand_dims(low_dose_data, axis=0)  # Convert to 4D tensor
        standard_dose_data = np.expand_dims(standard_dose_data, axis=0)  # Convert to 4D tensor
        mri_data = np.expand_dims(mri_data, axis=0)  # Convert to 4D tensor

        # Normalize data
        low_dose_data, low_dose_min, low_dose_max = normalize(low_dose_data)
        standard_dose_data, standard_dose_min, standard_dose_max = normalize(standard_dose_data)
        mri_data, mri_min, mri_max = normalize(mri_data)

        # Get the original height and width
        h, w = low_dose_data.shape[-2], low_dose_data.shape[-1]

        #Calculate the padding needed to reach 256x256
        pad_h = (256 - h) // 2
        pad_w = (256 - w) // 2
        pad_h_extra = (256 - h) % 2
        pad_w_extra = (256 - w) % 2

        low_dose_data = np.pad(low_dose_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)
        standard_dose_data = np.pad(standard_dose_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)
        mri_data = np.pad(mri_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)

        low_dose_data = torch.tensor(low_dose_data, dtype=torch.float32)
        standard_dose_data = torch.tensor(standard_dose_data, dtype=torch.float32)
        mri_data = torch.tensor(mri_data, dtype=torch.float32)

        if self.transform:
            low_dose_data = self.transform(low_dose_data)
            standard_dose_data = self.transform(standard_dose_data)
            mri_data = self.transform(mri_data)
        
        return low_dose_data, standard_dose_data, mri_data

class BiTaskPETDatasetMat_128(Dataset):
    def __init__(self, low_dose_dir, standard_dose_dir, mri_dir, transform=None):
        self.low_dose_files = sorted([os.path.join(low_dose_dir, f) for f in os.listdir(low_dose_dir) if f.endswith('.mat')])
        self.standard_dose_files = sorted([os.path.join(standard_dose_dir, f) for f in os.listdir(standard_dose_dir) if f.endswith('.mat')])
        self.mri_files = sorted([os.path.join(mri_dir, f) for f in os.listdir(mri_dir) if f.endswith('.mat')])
        self.transform = transform

    def __len__(self):
        return len(self.low_dose_files)

    def __getitem__(self, idx):
        low_dose_data = scipy.io.loadmat(self.low_dose_files[idx])['slice_data'] #[:80, :96]
        standard_dose_data = scipy.io.loadmat(self.standard_dose_files[idx])['slice_data'] #[:80, :96]
        mri_data = scipy.io.loadmat(self.mri_files[idx])['slice_data'] #[:80, :96]

        low_dose_data = np.expand_dims(low_dose_data, axis=0)  # Convert to 4D tensor
        standard_dose_data = np.expand_dims(standard_dose_data, axis=0)  # Convert to 4D tensor
        mri_data = np.expand_dims(mri_data, axis=0)  # Convert to 4D tensor

        # Normalize data
        low_dose_data, low_dose_min, low_dose_max = normalize(low_dose_data)
        standard_dose_data, standard_dose_min, standard_dose_max = normalize(standard_dose_data)
        mri_data, mri_min, mri_max = normalize(mri_data)

        # Get the original height and width
        h, w = low_dose_data.shape[-2], low_dose_data.shape[-1]

        #Calculate the padding needed to reach 256x256
        pad_h = (128 - h) // 2
        pad_w = (128 - w) // 2
        pad_h_extra = (128 - h) % 2
        pad_w_extra = (128 - w) % 2

        low_dose_data = np.pad(low_dose_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)
        standard_dose_data = np.pad(standard_dose_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)
        mri_data = np.pad(mri_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)

        low_dose_data = torch.tensor(low_dose_data, dtype=torch.float32)
        standard_dose_data = torch.tensor(standard_dose_data, dtype=torch.float32)
        mri_data = torch.tensor(mri_data, dtype=torch.float32)

        if self.transform:
            low_dose_data = self.transform(low_dose_data)
            standard_dose_data = self.transform(standard_dose_data)
            mri_data = self.transform(mri_data)
        
        return low_dose_data, standard_dose_data, mri_data

class BiTaskSegPETDatasetMat(Dataset):
    def __init__(self, low_dose_dir, standard_dose_dir, mri_dir, seg_mask_dir, transform=None):
        self.low_dose_files = sorted([os.path.join(low_dose_dir, f) for f in os.listdir(low_dose_dir) if f.endswith('.mat')])
        self.standard_dose_files = sorted([os.path.join(standard_dose_dir, f) for f in os.listdir(standard_dose_dir) if f.endswith('.mat')])
        self.mri_files = sorted([os.path.join(mri_dir, f) for f in os.listdir(mri_dir) if f.endswith('.mat')])
        self.seg_mask_files = sorted([os.path.join(seg_mask_dir, f) for f in os.listdir(seg_mask_dir) if f.endswith('.mat')])
        self.transform = transform

    def __len__(self):
        return len(self.low_dose_files)

    def __getitem__(self, idx):
        # Load data
        low_dose_data = scipy.io.loadmat(self.low_dose_files[idx])['slice_data']
        standard_dose_data = scipy.io.loadmat(self.standard_dose_files[idx])['slice_data']
        mri_data = scipy.io.loadmat(self.mri_files[idx])['slice_data']
        seg_mask = scipy.io.loadmat(self.seg_mask_files[idx])['slice_data']  # Assuming segmentation mask is named 'slice_data'

        # Expand dimensions for single-channel data
        low_dose_data = np.expand_dims(low_dose_data, axis=0)  # Shape: [1, H, W]
        standard_dose_data = np.expand_dims(standard_dose_data, axis=0)  # Shape: [1, H, W]
        mri_data = np.expand_dims(mri_data, axis=0)  # Shape: [1, H, W]

        # Normalize data
        low_dose_data, _, _ = normalize(low_dose_data)
        standard_dose_data, _, _ = normalize(standard_dose_data)
        mri_data, _, _ = normalize(mri_data)

        # Get the original height and width
        h, w = low_dose_data.shape[-2], low_dose_data.shape[-1]

        # Calculate the padding needed to reach 256x256
        pad_h = (256 - h) // 2
        pad_w = (256 - w) // 2
        pad_h_extra = (256 - h) % 2
        pad_w_extra = (256 - w) % 2

        # Apply padding
        low_dose_data = np.pad(low_dose_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)
        standard_dose_data = np.pad(standard_dose_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)
        mri_data = np.pad(mri_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)
        seg_mask = np.pad(seg_mask, ((pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)

        # Generate 3-channel T1 data using segmentation mask
        t1_channel_1 = np.where(seg_mask == 1, mri_data, 0)
        t1_channel_2 = np.where(seg_mask == 2, mri_data, 0)
        t1_channel_3 = np.where(seg_mask == 3, mri_data, 0)
        t1_3_channel = np.concatenate((t1_channel_1, t1_channel_2, t1_channel_3), axis=0)  # Shape: [3, H, W]

        # Convert to PyTorch tensors
        low_dose_data = torch.tensor(low_dose_data, dtype=torch.float32)
        standard_dose_data = torch.tensor(standard_dose_data, dtype=torch.float32)
        t1_3_channel = torch.tensor(t1_3_channel, dtype=torch.float32)

        # Apply transformations if provided
        if self.transform:
            low_dose_data = self.transform(low_dose_data)
            standard_dose_data = self.transform(standard_dose_data)
            t1_3_channel = self.transform(t1_3_channel)

        return low_dose_data, standard_dose_data, t1_3_channel

class BiTaskSegConcatPETDatasetMat(Dataset):
    def __init__(self, low_dose_dir, standard_dose_dir, mri_dir, seg_mask_dir, transform=None):
        self.low_dose_files = sorted([os.path.join(low_dose_dir, f) for f in os.listdir(low_dose_dir) if f.endswith('.mat')])
        self.standard_dose_files = sorted([os.path.join(standard_dose_dir, f) for f in os.listdir(standard_dose_dir) if f.endswith('.mat')])
        self.mri_files = sorted([os.path.join(mri_dir, f) for f in os.listdir(mri_dir) if f.endswith('.mat')])
        self.seg_mask_files = sorted([os.path.join(seg_mask_dir, f) for f in os.listdir(seg_mask_dir) if f.endswith('.mat')])
        self.transform = transform

    def __len__(self):
        return len(self.low_dose_files)

    def __getitem__(self, idx):
        # Load data
        print(self.low_dose_files[idx])
        low_dose_data = scipy.io.loadmat(self.low_dose_files[idx])['slice_data']
        standard_dose_data = scipy.io.loadmat(self.standard_dose_files[idx])['slice_data']
        mri_data = scipy.io.loadmat(self.mri_files[idx])['slice_data']
        seg_mask = scipy.io.loadmat(self.seg_mask_files[idx])['slice_data']  # Assuming segmentation mask is named 'slice_data'

        # Expand dimensions for single-channel data
        low_dose_data = np.expand_dims(low_dose_data, axis=0)  # Shape: [1, H, W]
        standard_dose_data = np.expand_dims(standard_dose_data, axis=0)  # Shape: [1, H, W]
        mri_data = np.expand_dims(mri_data, axis=0)  # Shape: [1, H, W]

        # Normalize data
        low_dose_data, _, _ = normalize(low_dose_data)
        standard_dose_data, _, _ = normalize(standard_dose_data)
        mri_data, _, _ = normalize(mri_data)

        # Get the original height and width
        h, w = low_dose_data.shape[-2], low_dose_data.shape[-1]

        # Calculate the padding needed to reach 256x256
        pad_h = (256 - h) // 2
        pad_w = (256 - w) // 2
        pad_h_extra = (256 - h) % 2
        pad_w_extra = (256 - w) % 2

        # Apply padding
        low_dose_data = np.pad(low_dose_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)
        standard_dose_data = np.pad(standard_dose_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)
        mri_data = np.pad(mri_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)
        seg_mask = np.pad(seg_mask, ((pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)

        # Generate 3-channel segmentation mask
        seg_channel_1 = (seg_mask == 1).astype(np.float32)
        seg_channel_2 = (seg_mask == 2).astype(np.float32)
        seg_channel_3 = (seg_mask == 3).astype(np.float32)
        seg_3_channel = np.stack((seg_channel_1, seg_channel_2, seg_channel_3), axis=0)  # Shape: [3, H, W]

        # Convert to PyTorch tensors
        low_dose_data = torch.tensor(low_dose_data, dtype=torch.float32)
        standard_dose_data = torch.tensor(standard_dose_data, dtype=torch.float32)
        mri_data = torch.tensor(mri_data, dtype=torch.float32)
        seg_3_channel = torch.tensor(seg_3_channel, dtype=torch.float32)

        # Apply transformations if provided
        if self.transform:
            low_dose_data = self.transform(low_dose_data)
            standard_dose_data = self.transform(standard_dose_data)
            mri_data = self.transform(mri_data)
            seg_3_channel = self.transform(seg_3_channel)

        return low_dose_data, standard_dose_data, mri_data, seg_3_channel
    
class BiTaskGMConcatPETDatasetMat(Dataset):
    def __init__(self, low_dose_dir, standard_dose_dir, mri_dir, seg_mask_dir, transform=None):
        self.low_dose_files = sorted([os.path.join(low_dose_dir, f) for f in os.listdir(low_dose_dir) if f.endswith('.mat')])
        self.standard_dose_files = sorted([os.path.join(standard_dose_dir, f) for f in os.listdir(standard_dose_dir) if f.endswith('.mat')])
        self.mri_files = sorted([os.path.join(mri_dir, f) for f in os.listdir(mri_dir) if f.endswith('.mat')])
        self.seg_mask_files = sorted([os.path.join(seg_mask_dir, f) for f in os.listdir(seg_mask_dir) if f.endswith('.mat')])
        self.transform = transform

    def __len__(self):
        return len(self.low_dose_files)

    def __getitem__(self, idx):
        # Load data
        print(self.low_dose_files[idx])
        low_dose_data = scipy.io.loadmat(self.low_dose_files[idx])['slice_data']
        standard_dose_data = scipy.io.loadmat(self.standard_dose_files[idx])['slice_data']
        mri_data = scipy.io.loadmat(self.mri_files[idx])['slice_data']
        seg_data = scipy.io.loadmat(self.seg_mask_files[idx])['slice_data']  # Assuming segmentation mask is named 'slice_data'
        filename = os.path.basename(self.low_dose_files[idx])

        # Expand dimensions for single-channel data
        low_dose_data = np.expand_dims(low_dose_data, axis=0)  # Shape: [1, H, W]
        standard_dose_data = np.expand_dims(standard_dose_data, axis=0)  # Shape: [1, H, W]
        mri_data = np.expand_dims(mri_data, axis=0)  # Shape: [1, H, W]
        seg_data = np.expand_dims(seg_data, axis=0)  # Shape: [1, H, W]

        # Normalize data
        low_dose_data, _, _ = normalize(low_dose_data)
        standard_dose_data, _, _ = normalize(standard_dose_data)
        mri_data, _, _ = normalize(mri_data)

        # Get the original height and width
        h, w = low_dose_data.shape[-2], low_dose_data.shape[-1]

        # Calculate the padding needed to reach 256x256
        pad_h = (256 - h) // 2
        pad_w = (256 - w) // 2
        pad_h_extra = (256 - h) % 2
        pad_w_extra = (256 - w) % 2

        # Apply padding
        low_dose_data = np.pad(low_dose_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)
        standard_dose_data = np.pad(standard_dose_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)
        mri_data = np.pad(mri_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)
        seg_data = np.pad(seg_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)

        # Convert to PyTorch tensors
        low_dose_data = torch.tensor(low_dose_data, dtype=torch.float32)
        standard_dose_data = torch.tensor(standard_dose_data, dtype=torch.float32)
        mri_data = torch.tensor(mri_data, dtype=torch.float32)
        seg_data = torch.tensor(seg_data, dtype=torch.float32)

        # Apply transformations if provided
        if self.transform:
            low_dose_data = self.transform(low_dose_data)
            standard_dose_data = self.transform(standard_dose_data)
            mri_data = self.transform(mri_data)
            seg_data = self.transform(seg_data)

        return low_dose_data, standard_dose_data, mri_data, seg_data, filename



# class PETDatasetBlackout(Dataset):
#     def __init__(self, low_dose_dir, standard_dose_dir, target_max=16384.0, transform=None):
#         self.low_dose_files = sorted([os.path.join(low_dose_dir, f) for f in os.listdir(low_dose_dir) if f.endswith('.mat')])
#         self.standard_dose_files = sorted([os.path.join(standard_dose_dir, f) for f in os.listdir(standard_dose_dir) if f.endswith('.mat')])
#         self.transform = transform
#         self.target_max = target_max

#     def __len__(self):
#         return len(self.low_dose_files)

#     def __getitem__(self, idx):
#         low_dose_data = scipy.io.loadmat(self.low_dose_files[idx])['slice_data'] #[:80, :96]
#         standard_dose_data = scipy.io.loadmat(self.standard_dose_files[idx])['slice_data'] #[:80, :96]

#         low_dose_data = np.expand_dims(low_dose_data, axis=0)  # Convert to 4D tensor
#         standard_dose_data = np.expand_dims(standard_dose_data, axis=0)  # Convert to 4D tensor

#         # Determine the global max across both low-dose and standard-dose data
#         global_max = max(low_dose_data.max(), standard_dose_data.max())

#         # Normalize both low-dose and standard-dose data based on the global max only
#         low_dose_data = self.normalize_to_255(low_dose_data, global_max, self.target_max).astype(np.int32)
#         standard_dose_data = self.normalize_to_255(standard_dose_data, global_max, self.target_max).astype(np.int32)


#         # Get the original height and width
#         h, w = low_dose_data.shape[-2], low_dose_data.shape[-1]

#         #Calculate the padding needed to reach 256x256
#         pad_h = (256 - h) // 2
#         pad_w = (256 - w) // 2
#         pad_h_extra = (256 - h) % 2
#         pad_w_extra = (256 - w) % 2

#         low_dose_data = np.pad(low_dose_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)
#         standard_dose_data = np.pad(standard_dose_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)

#         low_dose_data = torch.tensor(low_dose_data, dtype=torch.int32)
#         standard_dose_data = torch.tensor(standard_dose_data, dtype=torch.int32)

#         if self.transform:
#             low_dose_data = self.transform(low_dose_data)
#             standard_dose_data = self.transform(standard_dose_data)
        
#         return low_dose_data, standard_dose_data
    
#     def normalize_to_255(self, data, global_max, target_max):
#         """Normalize numpy array data to the range [0, 255] based on the global max only."""
#         # Avoid division by zero in case global_max is zero
#         if global_max > 0:
#             data = (data / global_max) * target_max
#         else:
#             data = np.zeros_like(data)
#         return data

class PETDatasetBlackout(Dataset):
    def __init__(self, low_dose_dir, standard_dose_dir, target_max=256.0, transform=None):
        self.low_dose_files = sorted([os.path.join(low_dose_dir, f) for f in os.listdir(low_dose_dir) if f.endswith('.mat')])
        self.standard_dose_files = sorted([os.path.join(standard_dose_dir, f) for f in os.listdir(standard_dose_dir) if f.endswith('.mat')])
        self.transform = transform
        self.target_max = target_max

    def __len__(self):
        return len(self.low_dose_files)

    def __getitem__(self, idx):
        low_dose_data = scipy.io.loadmat(self.low_dose_files[idx])['slice_data'] #[:80, :96]
        standard_dose_data = scipy.io.loadmat(self.standard_dose_files[idx])['slice_data'] #[:80, :96]

        low_dose_data = np.expand_dims(low_dose_data, axis=0)  # Convert to 4D tensor
        standard_dose_data = np.expand_dims(standard_dose_data, axis=0)  # Convert to 4D tensor

        # Determine the global max across both low-dose and standard-dose data
        global_max = max(low_dose_data.max(), standard_dose_data.max())

        # Normalize both low-dose and standard-dose data based on the global max only
        low_dose_data = self.normalize_to_255(low_dose_data, global_max, self.target_max)#.astype(np.int32)
        standard_dose_data = self.normalize_to_255(standard_dose_data, global_max, self.target_max)#.astype(np.int32)


        # Get the original height and width
        h, w = low_dose_data.shape[-2], low_dose_data.shape[-1]

        #Calculate the padding needed to reach 256x256
        pad_h = (256 - h) // 2
        pad_w = (256 - w) // 2
        pad_h_extra = (256 - h) % 2
        pad_w_extra = (256 - w) % 2

        low_dose_data = np.pad(low_dose_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)
        standard_dose_data = np.pad(standard_dose_data, ((0, 0), (pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)), mode='constant', constant_values=0)

        low_dose_data = torch.tensor(low_dose_data)#, dtype=torch.int32)
        standard_dose_data = torch.tensor(standard_dose_data)#, dtype=torch.int32)

        if self.transform:
            low_dose_data = self.transform(low_dose_data)
            standard_dose_data = self.transform(standard_dose_data)
        
        return {'ld':low_dose_data, 'sd':standard_dose_data}
    
    def normalize_to_255(self, data, global_max, target_max):
        """Normalize numpy array data to the range [0, 255] based on the global max only."""
        # Avoid division by zero in case global_max is zero
        if global_max > 0:
            data = (data / global_max) * target_max
        else:
            data = np.zeros_like(data)
        return data


#Updated normalize function to divide by 255
def normalize_png(image, max_value=255):
    image = image / max_value
    return image

class PETDatasetPNG(Dataset):
    def __init__(self, low_dose_dir, standard_dose_dir, transform=None):
        self.low_dose_files = sorted([os.path.join(low_dose_dir, f) for f in os.listdir(low_dose_dir) if f.endswith('.png')])
        self.standard_dose_files = sorted([os.path.join(standard_dose_dir, f) for f in os.listdir(standard_dose_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.low_dose_files)

    def __getitem__(self, idx):
        # Load PNG images
        low_dose_image = Image.open(self.low_dose_files[idx]).convert('L')
        standard_dose_image = Image.open(self.standard_dose_files[idx]).convert('L')
        
        # Convert to numpy array
        low_dose_data = np.expand_dims(low_dose_image, axis=0)  # Convert to 4D tensor
        standard_dose_data = np.expand_dims(standard_dose_image, axis=0)  # Convert to 4D tensor
        
        # Normalize the data by a fixed max value (255)
        low_dose_data = normalize_png(low_dose_data, max_value=255)
        standard_dose_data = normalize_png(standard_dose_data, max_value=255)
        
        # Convert to torch tensors
        low_dose_data = torch.tensor(low_dose_data, dtype=torch.float32)
        standard_dose_data = torch.tensor(standard_dose_data, dtype=torch.float32)
        
        # If transform is specified, apply it
        if self.transform:
            low_dose_data = self.transform(low_dose_data)
            standard_dose_data = self.transform(standard_dose_data)
        
        return low_dose_data, standard_dose_data


def read_sinogram(file_path, matrix_size=(344, 252, 4084), dtype=np.int16):
    """
    Reads the sinogram data from a binary file.

    Parameters:
        file_path (str): The path to the sinogram file.
        matrix_size (tuple): The dimensions of the sinogram data.
        dtype (numpy.dtype): The data type of the sinogram values.

    Returns:
        numpy.ndarray: The sinogram data as a NumPy array.
    """
    # Calculate the total number of elements in the sinogram
    total_elements = np.prod(matrix_size)

    # Read the binary data from the file
    with open(file_path, 'rb') as file:
        data = np.fromfile(file, dtype=dtype, count=total_elements)

    # Reshape the flat data into the specified matrix size
    sinogram = data.reshape(matrix_size)

    return sinogram

class PETDatasetSinoRaw(Dataset):
    def __init__(self, lowdose_dir, standarddose_dir, transform=None):
        self.lowdose_dir = Path(lowdose_dir)
        self.standarddose_dir = Path(standarddose_dir)
        self.transform = transform
        self.lowdose_files = sorted(self.lowdose_dir.glob('*'))
        self.standarddose_files = sorted(self.standarddose_dir.glob('*'))
        print(self.lowdose_files)

    def __len__(self):
        return len(self.lowdose_files)
    
    def __getitem__(self, idx):
        lowdose_path = self.lowdose_files[idx]
        standarddose_path = self.standarddose_files[idx]
        
        low_dose_data = read_sinogram(lowdose_path)
        standard_dose_data = read_sinogram(standarddose_path)
        
        # Apply log transform
        # low_dose_data = log_transform(low_dose_data)
        # standard_dose_data = log_transform(standard_dose_data)
        
        # Normalize data
        low_dose_data, low_dose_min, low_dose_max = normalize(low_dose_data)
        standard_dose_data, standard_dose_min, standard_dose_max = normalize(standard_dose_data)
        
        lowdose_tensor = torch.tensor(low_dose_data, dtype=torch.float32).unsqueeze(0)
        standarddose_tensor = torch.tensor(standard_dose_data, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            lowdose_tensor = self.transform(lowdose_tensor)
            standarddose_tensor = self.transform(standarddose_tensor)
        
        return lowdose_tensor, standarddose_tensor

class PETDatasetSino(Dataset):
    def __init__(self, low_dose_dir, standard_dose_dir, transform=None):
        self.low_dose_files = sorted([os.path.join(low_dose_dir, f) for f in os.listdir(low_dose_dir) if f.endswith('.mat')])
        self.standard_dose_files = sorted([os.path.join(standard_dose_dir, f) for f in os.listdir(standard_dose_dir) if f.endswith('.mat')])
        self.transform = transform

    def __len__(self):
        return len(self.low_dose_files)

    def __getitem__(self, idx):
        low_dose_data = scipy.io.loadmat(self.low_dose_files[idx])['sino']
        standard_dose_data = scipy.io.loadmat(self.standard_dose_files[idx])['sino']

        low_dose_data = np.expand_dims(low_dose_data, axis=0)  # Convert to 4D tensor
        standard_dose_data = np.expand_dims(standard_dose_data, axis=0)  # Convert to 4D tensor

        # Normalize data
        # low_dose_data, low_dose_min, low_dose_max = normalize(low_dose_data)
        # standard_dose_data, standard_dose_min, standard_dose_max = normalize(standard_dose_data)

        low_dose_data = torch.tensor(low_dose_data, dtype=torch.float32)
        standard_dose_data = torch.tensor(standard_dose_data, dtype=torch.float32)
        
        if self.transform:
            low_dose_data = self.transform(low_dose_data)
            standard_dose_data = self.transform(standard_dose_data)
        
        return low_dose_data, standard_dose_data
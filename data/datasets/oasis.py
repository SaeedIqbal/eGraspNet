import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from utils.preprocessing import preprocess_image

class OASISDataset(Dataset):
    """OASIS Brain MRI Dataset"""
    def __init__(self, root='/home/phd/datasets/oasis/', split='train'):
        self.root = root
        self.split = split
        
        # Load file lists
        if split == 'train':
            self.image_files = self._load_file_list('train_images.txt')[:255]
        elif split == 'val':
            self.image_files = self._load_file_list('val_images.txt')
        else:
            self.image_files = self._load_file_list('test_images.txt')[:150]
        
        self.seg_files = [f.replace('image', 'seg') for f in self.image_files]
    
    def _load_file_list(self, filename):
        with open(os.path.join(self.root, filename), 'r') as f:
            files = [line.strip() for line in f.readlines()]
        return files
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load images
        image_path = os.path.join(self.root, self.image_files[idx])
        seg_path = os.path.join(self.root, self.seg_files[idx])
        
        image = nib.load(image_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()
        
        # Preprocess
        image = preprocess_image(image, target_size=(160, 192, 224))
        seg = preprocess_image(seg, target_size=(160, 192, 224), is_seg=True)
        
        # Convert to tensor
        image = torch.from_numpy(image).float().unsqueeze(0)
        seg = torch.from_numpy(seg).long()
        
        return {'image': image, 'seg': seg, 'filename': self.image_files[idx]}
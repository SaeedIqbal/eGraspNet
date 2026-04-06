import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import Config
from data.datasets.oasis import OASISDataset
from data.datasets.lpba40 import LPBA40Dataset
from models.e_grasp_net import EGRASPNet
from losses.total_loss import TotalLoss
from train.trainer import Trainer
from utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train E-GRASP-Net')
    parser.add_argument('--config', type=str, default='configs/oasis_config.yaml')
    parser.add_argument('--dataset', type=str, choices=['oasis', 'lpba40'], default='oasis')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Logger
    logger = setup_logger(f'experiments/{args.dataset}/logs/train.log')
    
    # Dataset
    if args.dataset == 'oasis':
        train_dataset = OASISDataset(root=Config.OASIS_ROOT, split='train')
        val_dataset = OASISDataset(root=Config.OASIS_ROOT, split='val')
    else:
        train_dataset = LPBA40Dataset(root=Config.LPBA40_ROOT, split='train')
        val_dataset = LPBA40Dataset(root=Config.LPBA40_ROOT, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = EGRASPNet().to(device)
    
    # Loss
    criterion = TotalLoss(lambda_reg=Config.LAMBDA_REG).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # Trainer
    trainer = Trainer(model, criterion, optimizer, device, logger, args.dataset)
    
    # Training loop
    best_dice = 0.0
    for epoch in range(Config.NUM_EPOCHS):
        logger.info(f'Epoch {epoch+1}/{Config.NUM_EPOCHS}')
        
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss, val_dice = trainer.validate(val_loader, epoch)
        
        logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val DSC: {val_dice:.4f}')
        
        # Save checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
            }, f'experiments/{args.dataset}/checkpoints/best_model.pth')
        
        logger.info(f'Best DSC: {best_dice:.4f}')

if __name__ == '__main__':
    main()
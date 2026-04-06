import os

class Config:
    # Data
    DATA_ROOT = '/home/phd/datasets/'
    OASIS_ROOT = os.path.join(DATA_ROOT, 'oasis/')
    LPBA40_ROOT = os.path.join(DATA_ROOT, 'lpba40/')
    
    # Image dimensions
    IMG_SIZE = (160, 192, 224)
    VOXEL_SIZE = 1.0  # mm
    
    # Model
    ENCODER_CHANNELS = [32, 64, 128, 256, 512]
    DECODER_CHANNELS = [256, 128, 64, 32, 16]
    NUM_HEADS = 8
    HEAD_DIM = 64
    
    # Training
    BATCH_SIZE = 1
    NUM_EPOCHS = 500
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    LAMBDA_REG = 1.0
    
    # SPN
    SPN_START_LAYER = 3  # Start deformation estimation from layer 3
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Logging
    LOG_DIR = './experiments/'
    CHECKPOINT_DIR = './experiments/{}/checkpoints/'
    RESULT_DIR = './experiments/{}/results/'
import datasets
import model

from lightning import Trainer
from lightning.pytorch.callbacks import ModelSummary, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torchvision.datasets import CIFAR10, CIFAR100, Flowers102, OxfordIIITPet
import torch
from torchvision.models.vision_transformer import ViT_B_16_Weights
from torchvision import models

import wandb
import argparse
wandb.login()

def get_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--dataset', '-d', default='flowers102', type=str, help='Dataset')
    parser.add_argument('--epochs', '-e', default=20, type=int, help='Number of epochs')
    parser.add_argument('--model', '-m', default='vit', type=str, help='Model')
    parser.add_argument('--batch-size', '-b', dest='batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', default=0.00005, type=float, help='Learning rate')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'flowers102':
        DATASET = Flowers102
        N_CLASSES = 102
        IMG_SIZE = (224, 224)
        PATCH_SIZE = (16, 16)
    elif args.dataset == 'oxfordiiitpet':
        DATASET = OxfordIIITPet
        N_CLASSES = 37
        IMG_SIZE = (224, 224)
        PATCH_SIZE = (16, 16)
    elif args.dataset == 'cifar10':
        DATASET = CIFAR10
        N_CLASSES = 10
        IMG_SIZE = (32, 32)
        PATCH_SIZE = (4, 4)
    elif args.dataset == 'cifar100':
        DATASET = CIFAR100
        N_CLASSES = 100
        IMG_SIZE = (32, 32)
        PATCH_SIZE = (4, 4)
    else:
        raise Exception('No Dataset')


    if args.model == 'vit':
        MODEL = model.ViT(img_size=IMG_SIZE, patch_size=PATCH_SIZE, n_channels=3, 
                    nhead=4, dim_feedforward=1024, blocks=12, n_classes=N_CLASSES,
                    mlp_head_units=[1024, 512], d_model=512)
    elif args.model == 'pretrained-vit':
        MODEL = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        IMG_SIZE = (384, 384)
    else:
        raise Exception('No Model')
    

    wandb_logger = WandbLogger(project='ViT_test',
                               config={'batch_size': BATCH_SIZE, 'epochs': EPOCHS, 
                                       'learning_rate': LEARNING_RATE, 
                                       'img_size': IMG_SIZE, 'patch_size': PATCH_SIZE})
    trainer = Trainer(max_epochs=EPOCHS,
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      callbacks=[ModelSummary(max_depth=-1), EarlyStopping(monitor='val_loss_epoch')],
                      logger=wandb_logger)
    data_module = datasets.DataModule(IMAGE_SIZE=IMG_SIZE, batch_size=BATCH_SIZE, dataset=DATASET)
    train_model = model.ModelModule(model=MODEL, learning_rate=LEARNING_RATE)
    
    trainer.fit(train_model, data_module)


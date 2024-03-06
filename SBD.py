!pip install timm


import numpy as np
import pandas as pd
import os
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models
!pip install timm
import timm
from timm.loss import LabelSmoothingCrossEntropy # This is better than normal nn.CrossEntropyLoss

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
%matplotlib inline
import sys


import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision import models
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import train_test_split





!pip install --quiet pytorch-lightning>=1.4


import random
import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
%matplotlib inline

import math
import json
from functools import partial
from PIL import Image
import time

## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')
%matplotlib inline

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

## tqdm for loading bars
from tqdm.notebook import tqdm

## PyTorch

import torch.utils.data as data
import torch.optim as optim

## Torchvision
import torchvision
from torchvision import transforms

# Import tensorboard
%load_ext tensorboard
# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError: # Google Colab does not have PyTorch Lightning installed by default. Hence, we do it here if necessary

    import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

#Loading data into Transforms
from torchvision import datasets, transforms
from torch.utils.data import random_split
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop((224,224),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                      transforms.ToTensor()
                                     ])
satellite_dataset =  datasets.ImageFolder('/content/drive/MyDrive/Data/FloodPlains_7m/', transform=transform)
dataset_size = len(satellite_dataset)
test_split=0.2
test_size = int(test_split * dataset_size)
train_size = dataset_size - test_size

train_set, test_set = random_split(satellite_dataset,
                                               [train_size, test_size])


#Visualise multiple bands
from glob import glob
!pip install earthpy
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
import os
from rasterio.enums import Resampling



def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes
classes = get_classes("/content/drive/MyDrive/Data/FloodPlains_7m/")
classes

for dir in classes:
  print(dir,len(os.listdir("/content/drive/MyDrive/Data/FloodPlains_7m/"+dir)))

#Loading data into Transforms
from torchvision import datasets, transforms
from torch.utils.data import random_split
data_path = "/content/drive/MyDrive/Data/FloodPlains_7m/"
classes = ['Concrete', 'Silt', 'Vegetation', 'Water']
dataset_size = len(os.listdir(data_path+classes[0]))+ len(os.listdir(data_path+classes[1]))+ len(os.listdir(data_path+classes[2])) + len(os.listdir(data_path+classes[3]))

target = torch.zeros((dataset_size))
index = 0
for image in os.listdir(data_path+classes[0]):
  target[index] = 0
  index+=1
for image in os.listdir(data_path+classes[1]):
  target[index] = 1
  index+=1
for image in os.listdir(data_path+classes[2]):
  target[index] = 2
  index+=1
for image in os.listdir(data_path+classes[3]):
  target[index] = 3
  index+=1

#Labels
target_tens = torch.stack([target[i] for i in range(len(target))])

concrete_list = os.listdir(data_path+classes[0])
silt_list = os.listdir(data_path+classes[1])
vegetation_list = os.listdir(data_path+classes[2])
water_list = os.listdir(data_path+classes[3])

concrete_list = os.listdir(data_path+classes[0])
silt_list = os.listdir(data_path+classes[1])
vegetation_list = os.listdir(data_path+classes[2])
water_list = os.listdir(data_path+classes[3])

images = torch.zeros([dataset_size,16,2,2])
index = 0
for name in concrete_list:
  imagePath = data_path+classes[0]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(16,2,2),resampling=Resampling.bilinear)
  img_tens = torch.Tensor(image_ar)
  img_tensor_reshaped = img_tens.reshape(16*2,-1)
  tensor_reshaped = img_tensor_reshaped[~torch.all(img_tensor_reshaped.isnan(),dim=1)]
  img_tensor = tensor_reshaped.reshape(16,int(tensor_reshaped.shape[0]/16),-1)
  img_tens = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=[2,2], mode='bilinear').squeeze(0)
  img_tens = img_tens.transpose(dim0=1,dim1=2)
  img_tensor_reshaped = img_tens.reshape(16*2,-1)
  tensor_reshaped = img_tensor_reshaped[~torch.all(img_tensor_reshaped.isnan(),dim=1)]
  img_tensor = tensor_reshaped.reshape(16,int(tensor_reshaped.shape[0]/16),-1)
  img_tens = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=[2,2], mode='bilinear').squeeze(0)
  img_tens = img_tens.transpose(dim0=1,dim1=2)
  img_tensor_reshaped = img_tens.reshape(16*2,-1)
  tensor_reshaped = img_tensor_reshaped[~torch.any(img_tensor_reshaped.isnan(),dim=1)]
  img_tensor = tensor_reshaped.reshape(16,int(tensor_reshaped.shape[0]/16),-1)
  img_tens = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=[2,2], mode='bilinear').squeeze(0)
  images[index] = img_tens

  index+=1
for name in silt_list:
  imagePath = data_path+classes[1]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(16,2,2),resampling=Resampling.bilinear)
  img_tens = torch.Tensor(image_ar)
  img_tensor_reshaped = img_tens.reshape(16*2,-1)
  tensor_reshaped = img_tensor_reshaped[~torch.all(img_tensor_reshaped.isnan(),dim=1)]
  img_tensor = tensor_reshaped.reshape(16,int(tensor_reshaped.shape[0]/16),-1)
  img_tens = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=[2,2], mode='bilinear').squeeze(0)
  img_tens = img_tens.transpose(dim0=1,dim1=2)
  img_tensor_reshaped = img_tens.reshape(16*2,-1)
  tensor_reshaped = img_tensor_reshaped[~torch.all(img_tensor_reshaped.isnan(),dim=1)]
  img_tensor = tensor_reshaped.reshape(16,int(tensor_reshaped.shape[0]/16),-1)
  img_tens = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=[2,2], mode='bilinear').squeeze(0)
  img_tens = img_tens.transpose(dim0=1,dim1=2)
  img_tensor_reshaped = img_tens.reshape(16*2,-1)
  tensor_reshaped = img_tensor_reshaped[~torch.any(img_tensor_reshaped.isnan(),dim=1)]
  img_tensor = tensor_reshaped.reshape(16,int(tensor_reshaped.shape[0]/16),-1)
  img_tens = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=[2,2], mode='bilinear').squeeze(0)
  images[index] = img_tens
  index+=1
for name in vegetation_list:
  imagePath = data_path+classes[2]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(16,2,2),resampling=Resampling.bilinear)
  img_tens = torch.Tensor(image_ar)
  img_tensor_reshaped = img_tens.reshape(16*2,-1)
  tensor_reshaped = img_tensor_reshaped[~torch.all(img_tensor_reshaped.isnan(),dim=1)]
  img_tensor = tensor_reshaped.reshape(16,int(tensor_reshaped.shape[0]/16),-1)
  img_tens = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=[2,2], mode='bilinear').squeeze(0)
  img_tens = img_tens.transpose(dim0=1,dim1=2)
  img_tensor_reshaped = img_tens.reshape(16*2,-1)
  tensor_reshaped = img_tensor_reshaped[~torch.all(img_tensor_reshaped.isnan(),dim=1)]
  img_tensor = tensor_reshaped.reshape(16,int(tensor_reshaped.shape[0]/16),-1)
  img_tens = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=[2,2], mode='bilinear').squeeze(0)
  img_tens = img_tens.transpose(dim0=1,dim1=2)
  img_tensor_reshaped = img_tens.reshape(16*2,-1)
  tensor_reshaped = img_tensor_reshaped[~torch.any(img_tensor_reshaped.isnan(),dim=1)]
  img_tensor = tensor_reshaped.reshape(16,int(tensor_reshaped.shape[0]/16),-1)
  img_tens = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=[2,2], mode='bilinear').squeeze(0)
  images[index] = img_tens
  index+=1
for name in water_list:
  imagePath = data_path+classes[3]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(16,2,2),resampling=Resampling.bilinear)
  img_tens = torch.Tensor(image_ar)
  img_tensor_reshaped = img_tens.reshape(16*2,-1)
  tensor_reshaped = img_tensor_reshaped[~torch.all(img_tensor_reshaped.isnan(),dim=1)]
  img_tensor = tensor_reshaped.reshape(16,int(tensor_reshaped.shape[0]/16),-1)
  img_tens = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=[2,2], mode='bilinear').squeeze(0)
  img_tens = img_tens.transpose(dim0=1,dim1=2)
  img_tensor_reshaped = img_tens.reshape(16*2,-1)
  tensor_reshaped = img_tensor_reshaped[~torch.all(img_tensor_reshaped.isnan(),dim=1)]
  img_tensor = tensor_reshaped.reshape(16,int(tensor_reshaped.shape[0]/16),-1)
  img_tens = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=[2,2], mode='bilinear').squeeze(0)
  img_tens = img_tens.transpose(dim0=1,dim1=2)
  img_tensor_reshaped = img_tens.reshape(16*2,-1)
  tensor_reshaped = img_tensor_reshaped[~torch.any(img_tensor_reshaped.isnan(),dim=1)]
  img_tensor = tensor_reshaped.reshape(16,int(tensor_reshaped.shape[0]/16),-1)
  img_tens = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=[2,2], mode='bilinear').squeeze(0)
  images[index] = img_tens
  index+=1


#Image data stack
image_tens = torch.stack([images[i] for i in range(dataset_size)])

#Defining Custom dataset
my_dataset = torch.utils.data.TensorDataset(image_tens,target_tens)

val_split=0.2
val_size = int(val_split * dataset_size)
train_size = dataset_size - val_size

train_set, val_set = random_split(my_dataset, [train_size, val_size])

# We define a set of data loaders that we can use for various purposes later.
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, drop_last=False, pin_memory=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, drop_last=False, num_workers=4)

dataloaders = {
    "train": train_loader,
    "val": val_loader
}
train_data_len = len(train_set)
valid_data_len = len(val_set)
dataset_sizes = {
    "train": train_data_len,
    "val": valid_data_len
}
print(len(train_loader), len(val_loader))

print(train_data_len, valid_data_len)


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x

class AttentionBlock_0(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout)


    def forward(self, x):
        #x_1 = torch.roll(x,shifts=(1),dims=(1))
        #x_2 = torch.roll(x,shifts=(1),dims=(2))
        #inp_x = self.layer_norm_1(x)
        #inp_x1 = self.layer_norm_1(x_1)
        #inp_x2 = self.layer_norm_1(x_2)
        x = x + self.attn(x, x, x)[0]
        #x = x + self.linear(self.layer_norm_2(x))
        return x

class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        #x_1 = torch.roll(x,shifts=(1),dims=(1))
        #x_2 = torch.roll(x,shifts=(1),dims=(2))
        inp_x = self.layer_norm_1(x)
        #inp_x1 = self.layer_norm_1(x_1)
        #inp_x2 = self.layer_norm_1(x_2)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(x)
        #x = x + self.linear(self.layer_norm_2(x))
        return x

class VisionTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        #self.input_layer_0 = nn.Linear(num_channels, 3)
        self.transformer_0 = nn.Sequential(*[AttentionBlock_0(num_channels, embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)])

        self.input_layer = nn.Linear(num_channels, embed_dim)
        #self.input_layer = nn.Conv2d(16,embed_dim,kernel_size=3, padding=1)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))


    def forward(self, x):
        # Preprocess input
        #print("In VT",x.shape)

        x = x.permute(0,2,3,1)
        x_attn= self.transformer_0(x.squeeze(0))
        #print(x.shape, x_attn.shape)
        x = x + x_attn.unsqueeze(0)
        x = x.permute(0,3,1,2)

        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        #x = self.input_layer(x.permute(2,0,1)).permute(1,2,0)
        #x = self.input_layer_0(x)
        #x = x + self.in_attn(x,x,x)[0]
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        #print(x.shape, self.pos_embedding[:,:T+1].shape)
        #x = x + self.pos_embedding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        #print("Transformer out",x.shape)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        #print("final out",out)
        return out

from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix

class ViT(pl.LightningModule):

    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)
        self.example_input_array = next(iter(train_loader))[0]
        self.val_step_outputs = []        # save outputs in each batch to compute metric overall epoch
        self.val_step_targets = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        #loss = F.cross_entropy(preds, labels)
        #print("out shape", preds.shape, preds)
        label_one_hot = torch.nn.functional.one_hot(labels.type(torch.int64), num_classes = len(classes))
        loss = F.cross_entropy(preds[0], label_one_hot.type(torch.float32)[0].to(device))
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_acc', acc, prog_bar=True, on_step=False, on_epoch=True)

        if mode=='test':
          self.val_step_outputs.append(torch.tensor(preds.argmax(dim=-1), dtype=torch.int8))
          self.val_step_targets.append(torch.tensor(labels, dtype=torch.int8))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    def on_test_epoch_end(self):
        val_all_outputs = torch.stack(self.val_step_outputs)
        val_all_targets = torch.stack(self.val_step_targets)
        #print(len(val_all_outputs), len(val_all_targets))
        f1_macro_epoch = f1_score(val_all_outputs.cpu(), val_all_targets.cpu(),average='macro')
        precision_macro_epoch = precision_score(val_all_outputs.cpu(), val_all_targets.cpu(),average='macro')
        recall_macro_epoch = recall_score(val_all_outputs.cpu(), val_all_targets.cpu(),average='macro')
        cm_macro_epoch = confusion_matrix(val_all_outputs.cpu(), val_all_targets.cpu())
        print("Confusion Matrix", cm_macro_epoch)
        self.log(f'F1', f1_macro_epoch)
        self.log(f'Precision', precision_macro_epoch)
        self.log(f'Recall', recall_macro_epoch)
        self.val_step_outputs.clear()
        self.val_step_targets.clear()


def train_model(**kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "ViT"),
                         max_epochs = 180,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = True # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ViT.ckpt")
    if os.path.isfile(pretrained_filename):
        #print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = ViT.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(42) # To be reproducable
        model = ViT(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation and test set
    #train_result = trainer.test(model, train_loader, verbose=True)
    val_result = trainer.test(model, val_loader, verbose=True)
    #test_result = trainer.test(model, test_loader, verbose=False)
    result = {"val": val_result[0]["test_acc"]}

    return model, result

CHECKPOINT_PATH='/content/drive/MyDrive/'
model, results = train_model(model_kwargs={
                                'embed_dim': 256,
                                'hidden_dim': 512,
                                'num_heads': 8,
                                'num_layers': 1,
                                'patch_size': 1,
                                'num_channels': 16,
                                'num_patches': 4,
                                'num_classes': len(classes),
                                'dropout': 0.2
                            },
                            lr=3e-4)
print("ViT results", results)

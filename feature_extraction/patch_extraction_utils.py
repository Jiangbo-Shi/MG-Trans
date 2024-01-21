### Dependencies
# Base Dependencies
import os

# LinAlg / Stats / Plotting Dependencies
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
from PIL import Image
from torch import nn
from tqdm import tqdm

# Torch Dependencies
import torch
import torch.multiprocessing
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision import transforms
device = torch.device('cuda')
torch.multiprocessing.set_sharing_strategy('file_system')

# Model Architectures
from resnet_trunc import resnet50_trunc_baseline


### Helper Functions for Normalization + Loading in pytorch_lightning SSL encoder (for SimCLR)

def eval_transforms(pretrained=False):
    if pretrained:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
    trnsfrms_val = transforms.Compose([transforms.ToTensor(), 
                                       transforms.Normalize(mean = mean, std = std)])
    return trnsfrms_val


def torchvision_ssl_encoder(name: str, pretrained: bool = False, return_all_feature_maps: bool = False):
    pretrained_model = getattr(resnets, name)(pretrained=pretrained, return_all_feature_maps=return_all_feature_maps)
    pretrained_model.fc = Identity()
    return pretrained_model


### Functions for Loading + Saving + Visualizing Patch Embeddings
def save_embeddings(model, fname, dataloader, enc_name, overwrite=False):

    if os.path.isfile('%s.h5' % fname) and (overwrite == False):
        return None

    embeddings, coords, file_names = [], [], []

    for batch, coord in dataloader:
        with torch.no_grad():
            # patch embeddings
            batch = batch.to(device)
            embeddings.append(model(batch).detach().cpu().numpy().squeeze())
            file_names.append(coord)

    for file_name in file_names:
        for coord in file_name:
            coord = coord.rstrip('.png').split('_')
            coords.append([int(coord[0]), int(coord[1])])

    print(fname)

    embeddings = np.vstack(embeddings)
    coords = np.vstack(coords)

    f = h5py.File(fname+'.h5', 'w')
    f['features'] = embeddings
    f['coords'] = coords
    f.close()


def create_embeddings(embeddings_dir, enc_name, dataset, batch_size, save_patches=False,
                      patch_datasets='path/to/patch/datasets', assets_dir ='./ckpts/',
                      disentangle=-1, stage=-1):
    print("Extracting Features for '%s' via '%s'" % (dataset, enc_name))
    if enc_name == 'resnet50_trunc':
        model = resnet50_trunc_baseline(pretrained=True)
        eval_t = eval_transforms(pretrained=True)
    
    else:
        pass

    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    # if 'simclr' in enc_name or 'simsiam' in enc_name:
    #     _model = model
    #     model = lambda x: _model.forward(x)[0]
    if 'dino' in enc_name:
        _model = model
        if stage == -1:
            model = _model
        else:
            model = lambda x: torch.cat([x[:, 0] for x in _model.get_intermediate_layers(x, stage)], dim=-1)

    if stage != -1:
        _stage = '_s%d' % stage
    else:
        _stage = ''

    if dataset == 'BRIGHT':
        # pool = ThreadPoolExecutor(max_workers=48)
        for wsi_name in tqdm(os.listdir(patch_datasets)):
            dataset = PatchesDataset(os.path.join(patch_datasets, wsi_name), transform=eval_t)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            fname = os.path.join(embeddings_dir, wsi_name)
            if(not os.path.exists(fname)):
                save_embeddings(model, fname, dataloader, enc_name)

                # args = [model, fname, dataloader]
                # pool.submit(lambda p: save_embeddings(*p), args)
        # pool.shutdown(wait=True)


class PatchesDataset(Dataset):
    def __init__(self, file_path, transform=None):
        file_names = os.listdir(file_path)
        imgs = []
        coords = []
        for file_name in file_names:
            imgs.append(os.path.join(file_path, file_name))
            coords.append(file_name)
        self.imgs = imgs
        self.coords = coords
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        coord = self.coords[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, coord

    def __len__(self):
        return len(self.imgs)
        

import os

import openslide
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import h5py

# Define the following paths by yourself.
TG_5x_folder = 'patch of 5x h5 feature folders'
TG_10x_folder = 'patch of 10x h5 feature folders'
save_folder_10x = 'attn_mask save path'
slide_folder = 'WSI files path'

if(not os.path.exists(save_folder_10x)):
    os.makedirs(save_folder_10x)

all_data = np.array(pd.read_excel('data_BRIGHT_three.xlsx', engine='openpyxl',  header=None))
svs2uuid = {}
for i in all_data:
    svs2uuid[i[1]] = i[0]

def find_alignment(file_name):
    TG_5x_file_path = os.path.join(TG_5x_folder, file_name)
    TG_10x_file_path = os.path.join(TG_10x_folder, file_name)

    TG_5x = np.array(h5py.File(TG_5x_file_path, 'r')['coords']).astype(np.int)
    TG_10x = np.array(h5py.File(TG_10x_file_path, 'r')['coords']).astype(np.int)

    slide_path = os.path.join(slide_folder, svs2uuid[file_name.replace('h5', 'svs')], file_name.replace('h5', 'svs'))
    slide = openslide.open_slide(slide_path)
    shape = slide.level_dimensions[0]
    TG_5x_matrix = np.zeros((int(shape[0]/16), int(shape[1]/16)))
    TG_10x_matrix = np.zeros((int(shape[0]/16), int(shape[1]/16)))

    for i, coord in enumerate(TG_5x):
        TG_5x_matrix[int(coord[0]/16):int((coord[0]+2048)/16), int(coord[1]/16):int((coord[1]+2048)/16)] = i + 1
    for i, coord in enumerate(TG_10x):
        TG_10x_matrix[int(coord[0]/16):int((coord[0]+1024)/16), int(coord[1]/16):int((coord[1]+1024)/16)] = i + 1

    align_matrix_10x = torch.ones(TG_5x.shape[0], np.array(h5py.File(TG_10x_file_path, 'r')['coords']).shape[0])

    for i in tqdm(range(1, TG_5x.shape[0]+1)):
        cover_patches_10x = np.unique(TG_10x_matrix[TG_5x_matrix == i])
        cover_ids_10x = np.delete(cover_patches_10x, np.where(cover_patches_10x==0)) - 1
        align_matrix_10x[i-1][cover_ids_10x] = 0

    align_matrix_10x = align_matrix_10x.type(torch.bool)
    torch.save(~align_matrix_10x, os.path.join(save_folder_10x, file_name.replace('h5', 'pt')))


pool = ThreadPoolExecutor(max_workers=48)
for file_name in os.listdir(TG_5x_folder):
    if(file_name.replace('h5', 'pt') in os.listdir(save_folder_10x)):
        print(file_name, ': have been processed!')
    else:
        pool.submit(find_alignment, file_name)
pool.shutdown(wait=True)

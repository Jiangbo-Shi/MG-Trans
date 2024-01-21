import os
import argparse
import torch.multiprocessing
# device = torch.device('cuda:1')
torch.multiprocessing.set_sharing_strategy('file_system')
from patch_extraction_utils import create_embeddings
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Configurations for feature extraction')
parser.add_argument('--patches_path', type=str)
parser.add_argument('--library_path', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--batch_size', type=int)
args = parser.parse_args()

patches_path = args.patches_path
library_path = args.library_path
model_name = args.model_name
os.makedirs(library_path, exist_ok=True)

# enc_name: model name
# resnet50_trunc
create_embeddings(patch_datasets=patches_path, embeddings_dir=library_path,
                  enc_name=model_name, dataset='BRIGHT', batch_size=args.batch_size)
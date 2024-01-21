import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str)
parser.add_argument('--val_frac', type=float, default= 0.2,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.2,
                    help='fraction of labels for test (default: 0.1)')
parser.add_argument('--dataset', type=str, default='Yifuyuan')

args = parser.parse_args()

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/TCGA_RCC_subtyping.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'CCRCC':0, 'PRCC':1, 'CRCC':2},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])
elif args.task == 'task_subtyping_brca':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/TCGA_BRCA_subtyping.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'BRDC':0, 'BRLC':1},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])

elif args.task == 'task_gene_mutuation':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/TCGA_BRCA_MAP3K1.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'normal':0, 'MAP3K1':1},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])
elif args.task == 'task_gene_mutuation_MSIMSS':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/TCGA_STAD_MSIMSS.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'mss':0, 'msimut':1},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])
elif args.task == 'task_BRIGHT_cls3':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/BRIGHT_subtyping_three.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'Non-cancerous':0, 'Cancerous':1, 'Pre-cancerous':2},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])

elif args.task == 'task_BRIGHT_cls6':
    args.n_classes=6
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/BRIGHT_subtyping_six.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'PB':0, 'UDH':1, 'FEA':2, 'ADH':3, 'DCIS':4, 'IC':5},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])

elif args.task == 'task_3_pt_staging_cls3':
    args.n_classes = 3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/TCGA_gastric_cancer_pt_staging.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'T1ab':0, 'T2':1, 'T34':2},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])
elif args.task == 'task_3_pt_staging_cls2':
    args.n_classes = 2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/TCGA_BLCA_pt_staging.csv',
                                                 shuffle = False,
                                                 seed = args.seed,
                                                 print_info = True,
                                                 label_dict = {'T2':0, 'T3':1},
                                                 patient_strat= True,
                                                 patient_voting='maj',
                                                 ignore=[])

else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/'+ args.dataset +'/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))




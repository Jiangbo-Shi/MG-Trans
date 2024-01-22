export CUDA_VISIBLE_DEVICES=0

python patch_extraction.py \
--patches_path '/data_prepare/BRIGHT/patches_512' \
--library_path 'patch_feature_save_path' \
--model_name 'resnet50_trunc' \
--batch_size 64 \

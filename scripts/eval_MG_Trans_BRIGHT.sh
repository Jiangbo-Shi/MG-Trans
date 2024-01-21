export CUDA_VISIBLE_DEVICES=0
python eval.py \
--drop_out \
--k 5 \
--k_start 0 \
--k_end 5 \
--models_exp_code MG_Trans_s0 \
--save_exp_code MG_Trans_s0 \
--task task_BRIGHT_cls3 \
--model_type MG_Trans \
--mode transformer \
--results_dir results/BRIGHT \
--splits_dir splits/BRIGHT/task_BRIGHT_cls3_100 \
--data_root_dir DATA_ROOT_DIR/BRIGHT/resnet50_trunc \
--data_folder_s bright_subtyping_5x \
--data_folder_l bright_subtyping_10x \
--img_size 2500 \
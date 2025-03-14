python main.py \
        --device cuda \
        --batch_size 16  \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.0 \
        --dropout 0.0 \
        --lr_start 1e-6 \
        --num_workers 20\
        --max_epochs 1000 \
        --num_feats 32 \
        --checkpoint_every 3 \
        --seed_path './Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt' \
        --dataset_name FD_A0 \
        --data_root ./data/FD_A0 \
        --measure_name CCS \
        --checkpoints_folder './Linear_Allccs'\
        --project_name 'Linear_A'
 
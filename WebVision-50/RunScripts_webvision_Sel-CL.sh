CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_webvision_Sel-CL.py --epoch 130 --num_classes 50 --batch_size 64 --low_dim 128 --lr-scheduler "step"  \
--network "RN18" --lr 0.1 --wd 1e-4 --dataset "webvison" \
--sup_t 0.1 --headType "Linear"  --sup_queue_use 1 --sup_queue_begin 3 --queue_per_class 200 \
--alpha 0.4 --k_val 250 --trainval_root  /home/lishikun/data/mini-WebVision/ --val_root /home/lishikun/data/ImageNet/val/ --out ./out/ \
--experiment_name webvision_sup --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--uns_t 0.1 --uns_queue_k 10000 --lr-warmup-epoch 5 --warmup-epoch 40 --lambda_s 0.01 --lambda_c 1 \
--test_batch_size 100 --beta 0.0 --warmup_way 'sup'

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_webvision_Sel-CL.py --epoch 130 --num_classes 50 --batch_size 64 --low_dim 128 --lr-scheduler "step"  \
--network "RN18" --lr 0.1 --wd 1e-4 --dataset "webvison" \
--sup_t 0.1 --headType "Linear" --sup_queue_use 1 --sup_queue_begin 3 --queue_per_class 200 \
--alpha 0.4 --k_val 250 --trainval_root  /home/lishikun/data/mini-WebVision/ --val_root /home/lishikun/data/ImageNet/val/ --out ./out/ \
--experiment_name webvision_uns  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--uns_t 0.1 --uns_queue_k 10000 --lr-warmup-epoch 5 --warmup-epoch 5 --lambda_s 0.01 --lambda_c 1 \
--test_batch_size 100 --beta 0.0 --warmup_way 'uns'
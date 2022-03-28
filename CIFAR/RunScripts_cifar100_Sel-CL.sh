python3 train_Sel-CL.py --epoch 250 --num_classes 100 --batch_size 128 --low_dim 128 --lr-scheduler "step"  --noise_ratio 0.2 \
--network "PR18" --lr 0.1 --wd 1e-4 --dataset "CIFAR-100" --download True --noise_type "symmetric" \
--sup_t 0.1 --headType "Linear"  --sup_queue_use 1 --sup_queue_begin 3 --queue_per_class 100 \
--alpha 0.75 --beta 0.35 --k_val 250 --experiment_name CIFAR100 --cuda_dev 0  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--uns_t 0.1 --uns_queue_k 10000 --lr-warmup-epoch 5 --warmup-epoch 1  --lambda_s 0.01 --lambda_c 1 --warmup_way "uns"

python3 train_Sel-CL.py --epoch 250 --num_classes 100 --batch_size 128 --low_dim 128 --lr-scheduler "step"  --noise_ratio 0.4 \
--network "PR18" --lr 0.1 --wd 1e-4 --dataset "CIFAR-100" --download True --noise_type "asymmetric" \
--sup_t 0.1 --headType "Linear" --sup_queue_use 1 --sup_queue_begin 3 --queue_per_class 100 \
--alpha 0.25 --beta 0.0 --k_val 250 --experiment_name CIFAR100 --cuda_dev 0  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--uns_t 0.1 --uns_queue_k 10000 --lr-warmup-epoch 5 --warmup-epoch 1  --lambda_s 0.01 --lambda_c 1 --warmup_way "uns"

python3 train_Sel-CL_v2.py --epoch 250 --num_classes 100 --batch_size 128 --low_dim 128 --lr-scheduler "step"  --noise_ratio 0.2 \
--network "PR18" --lr 0.1 --wd 1e-4 --dataset "CIFAR-100" --download True --noise_type "symmetric" \
--sup_t 0.1 --headType "Linear" --sup_queue_use 1 --sup_queue_begin 3 --queue_per_class 100 \
--alpha 0.75 --beta 0.35 --k_val 250 --experiment_name CIFAR100_v2 --cuda_dev 0  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--uns_t 0.1 --uns_queue_k 10000 --lr-warmup-epoch 5 --warmup-epoch 1  --lambda_s 0.01 --lambda_c 1 --warmup_way "uns"

python3 train_Sel-CL_v2.py --epoch 250 --num_classes 100 --batch_size 128 --low_dim 128 --lr-scheduler "step"  --noise_ratio 0.4 \
--network "PR18" --lr 0.1 --wd 1e-4 --dataset "CIFAR-100" --download True --noise_type "asymmetric" \
--sup_t 0.1 --headType "Linear"  --sup_queue_use 1 --sup_queue_begin 3 --queue_per_class 100 \
--alpha 0.25 --beta 0.0 --k_val 250 --experiment_name CIFAR100_v2 --cuda_dev 0  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--uns_t 0.1 --uns_queue_k 10000 --lr-warmup-epoch 5 --warmup-epoch 1  --lambda_s 0.01 --lambda_c 1 --warmup_way "uns"
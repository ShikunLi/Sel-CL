python3 train_Sel-CL_fine-tuning.py --epoch 70 --num_classes 100 --batch_size 128 --noise_ratio 0.2 \
--network "PR18" --lr 0.001 --wd 1e-4 --dataset "CIFAR-100" --cuda_dev 0 \
--headType "Linear" --noise_type "symmetric" --DA "simple" --ReInitializeClassif 1  \
--startLabelCorrection 30  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--experiment_name CIFAR100 --train_root ./dataset --out ./out

python3 train_Sel-CL_fine-tuning.py --epoch 70 --num_classes 100 --batch_size 128  --noise_ratio 0.4 \
--network "PR18" --lr 0.001 --wd 1e-4 --dataset "CIFAR-100" --cuda_dev 0 \
--headType "Linear" --noise_type "asymmetric" --DA "simple" --ReInitializeClassif 1  \
--startLabelCorrection 30  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--experiment_name CIFAR100 --train_root ./dataset --out ./out


python3.8 train_Sel-CL_fine-tuning.py --epoch 70 --num_classes 100 --batch_size 128 --noise_ratio 0.2 \
--network "PR18" --lr 0.001 --wd 1e-4 --dataset "CIFAR-100" --cuda_dev 5 \
--headType "Linear" --noise_type "symmetric" --DA "simple" --ReInitializeClassif 1  \
--startLabelCorrection 30  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--experiment_name CIFAR100 --train_root ./dataset --out ./out


python3.8 train_Sel-CL_fine-tuning.py --epoch 70 --num_classes 100 --batch_size 128 --noise_ratio 0.4 \
--network "PR18" --lr 0.001 --wd 1e-4 --dataset "CIFAR-100" --cuda_dev 4 \
--headType "Linear" --noise_type "asymmetric" --DA "simple" --ReInitializeClassif 1  \
--startLabelCorrection 30  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--experiment_name CIFAR100_v2 --train_root ./dataset --out ./out
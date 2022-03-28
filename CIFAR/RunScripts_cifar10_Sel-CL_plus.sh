python3 train_Sel-CL_fine-tuning.py --epoch 70 --num_classes 10 --batch_size 128  --noise_ratio 0.2 \
--network "PR18" --lr 0.001 --wd 1e-4 --dataset "CIFAR-10" --cuda_dev 0 \
--headType "Linear" --noise_type "symmetric" --DA "Simple" --ReInitializeClassif 1  \
--startLabelCorrection 30  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--experiment_name CIFAR10 --train_root ./dataset --out ./out

python3 train_Sel-CL_fine-tuning.py --epoch 70 --num_classes 10 --batch_size 128 --noise_ratio 0.4 \
--network "PR18" --lr 0.001 --wd 1e-4 --dataset "CIFAR-10" --cuda_dev 0 \
--headType "Linear" --noise_type "asymmetric" --DA "Simple" --ReInitializeClassif 1  \
--startLabelCorrection 30  --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--experiment_name CIFAR10 --train_root ./dataset --out ./out
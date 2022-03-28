CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_webvision_Sel-CL_fine-tuning.py --epoch 50 --num_classes 50 --batch_size 64 \
--network "RN18" --lr 0.001 --wd 1e-4 --dataset "webvision" --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--headType "Linear" --ReInitializeClassif 1 --startLabelCorrection 20 --experiment_name webvision_sup \
--trainval_root  /home/lishikun/data/mini-WebVision/ --val_root /home/lishikun/data/ImageNet/val/ --out ./out/


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_webvision_Sel-CL_fine-tuning.py --epoch 50 --num_classes 50 --batch_size 64 \
--network "RN18" --lr 0.001 --wd 1e-4 --dataset "webvision" --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--headType "Linear" --ReInitializeClassif 1 --startLabelCorrection 20 --experiment_name webvision_uns \
--trainval_root  /home/lishikun/data/mini-WebVision/ --val_root /home/lishikun/data/ImageNet/val/ --out ./out/
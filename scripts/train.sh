python scripts/train.py \
--cfg scripts/configs/config.yml \
--cfg_munit ./munit/gta2cityscapes.yaml \
--tensorboard \
--exp-suffix 'test' \
--data-root /media/echiou/DATA/Documents/gta_cityscapes/ \
--project-root /home/echiou/Dropbox/PhD/code/Beyond-deterministic-translation-for-UDA \
--source-domain gta \
--num-classes 19 \
--train-restore-from /home/echiou/Dropbox/PhD/code/CVPR_2021/ADVENT_CVPR_21/pretrained_models/DeepLab_resnet_pretrained_imagenet.pth \
--use-synth-s2t True \
--transl-net munit \
--load-dir-transl /home/echiou/Dropbox/PhD/code/CVPR_2021/ADVENT_CVPR_21/pretrained_models/munit_gta_4_gpus_guide_transl_conf \
--load-iter-transl 150000 \
--batch-size 1 \
--gpu-ids '0' \
--subnorm_type batch \
--data_aug '' \
--var 1 \
--end_warm_up_iter 0 \
--use-synth-t2s '' \
--seed 1234 \
--use_ps True \
--round 4 \
--start-iter 102000 \
python scripts/test.py \
--cfg scripts/configs/config.yml \
--exp-suffix '' \
--data-root path/to/dataset \
--project-root path/to/code \
--source-domain gta \
--num-classes 19 \
--use-synth-t2s True \
--test-snapshot "/path/to/the/pretrained/source/segmentation/model" "/path/to/the/pretrained/target1/segmentation/model" "/path/to/the/pretrained/target2/segmentation/model" \
--load-dir-transl  /path/to/the/pretrained/image2image/translation/model \
--load-iter-transl 140000 \
--cfg_munit ./munit/gta2cityscapes.yaml  \
--num-fake-source 10 \
--gen-pseudo-labels True \
--pseudo-labels-path /path/to/store/pseudolabels

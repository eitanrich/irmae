#!/bin/bash
. /cs/labs/yweiss/eitanrich/eitan-venv2/bin/activate
python3 train.py --gpu --epochs 80 --dataset celeba --data-path ./data/stylegan_2_z100 -n 16 -l 8 --model-name irmae_l16_100
python3 embedding.py --dataset celeba --data-path ./data/stylegan_2_z100 -n 16 -l 8 --model-name irmae_l16_100

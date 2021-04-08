#!/bin/bash
. /cs/labs/yweiss/eitanrich/eitan-venv2/bin/activate
python3 train.py --gpu --epochs 100 --dataset celeba --data-path ./data/stylegan_2_z100 -n 2 --model-name ae_2_100
python3 embedding.py --dataset celeba --data-path ./data/stylegan_2_z100 -n 2 --model-name ae_2_100

#!/bin/bash
. /cs/labs/yweiss/eitanrich/eitan-venv2/bin/activate
echo "Training AE for $1"
python3 train.py --gpu --epochs 100 --dataset celeba --data-path /cs/labs/yweiss/shared/datasets/GANSpace/$1 -n 4 --model-name ae_$1
python3 embedding.py --dataset celeba --data-path /cs/labs/yweiss/shared/datasets/GANSpace/$1 -n 4 --model-name ae_$1

echo "Training IRMAE for $1"
python3 train.py --gpu --epochs 100 --dataset celeba --data-path /cs/labs/yweiss/shared/datasets/GANSpace/$1 -n 4 -l 8 --model-name irmae_$1
python3 embedding.py --dataset celeba --data-path /cs/labs/yweiss/shared/datasets/GANSpace/$1 -n 4 -l 8 --model-name irmae_$1

echo "Training VAE for $1"
python3 train.py --gpu --epochs 100 --dataset celeba --data-path /cs/labs/yweiss/shared/datasets/GANSpace/$1 -n 4 --vae --model-name vae_$1
python3 embedding.py --dataset celeba --data-path /cs/labs/yweiss/shared/datasets/GANSpace/$1 -n 4 --vae --model-name vae_$1

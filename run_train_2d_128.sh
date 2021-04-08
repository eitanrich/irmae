#!/bin/bash
. /cs/labs/yweiss/eitanrich/eitan-venv2/bin/activate
echo "Training AE for $1"
python3 train.py --gpu --epochs 100 --dataset celeba --data-path /cs/labs/yweiss/shared/datasets/GANSpace/$1 -n 128 --model-name ae_128_$1
python3 embedding.py --dataset celeba --data-path /cs/labs/yweiss/shared/datasets/GANSpace/$1 -n 128 --model-name ae_128_$1

#echo "Training IRMAE for $1"
#python3 train.py --gpu --epochs 100 --dataset celeba --data-path /cs/labs/yweiss/shared/datasets/GANSpace/$1 -n 128 -l 8 --model-name irmae_128_$1
#python3 embedding.py --dataset celeba --data-path /cs/labs/yweiss/shared/datasets/GANSpace/$1 -n 128 -l 8 --model-name irmae_128_$1

#echo "Training VAE for $1"
#python3 train.py --gpu --epochs 100 --dataset celeba --data-path /cs/labs/yweiss/shared/datasets/GANSpace/$1 -n 128 --vae --model-name vae_128_$1
#python3 embedding.py --dataset celeba --data-path /cs/labs/yweiss/shared/datasets/GANSpace/$1 -n 128 --vae --model-name vae_128_$1

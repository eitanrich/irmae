#!/bin/bash
sbatch --killable --mem=16g -c2 --gres=gpu:1,vmem:8g --time=1-0 run_train_2d_128.sh manifold_bedrooms_5_9
sbatch --killable --mem=16g -c2 --gres=gpu:1,vmem:8g --time=1-0 run_train_2d_128.sh manifold_car_0_13
sbatch --killable --mem=16g -c2 --gres=gpu:1,vmem:8g --time=1-0 run_train_2d_128.sh manifold_cat_4_13
sbatch --killable --mem=16g -c2 --gres=gpu:1,vmem:8g --time=1-0 run_train_2d_128.sh manifold_cat_6_9
sbatch --killable --mem=16g -c2 --gres=gpu:1,vmem:8g --time=1-0 run_train_2d_128.sh manifold_church_3_9
sbatch --killable --mem=16g -c2 --gres=gpu:1,vmem:8g --time=1-0 run_train_2d_128.sh manifold_ffhq_0_1
sbatch --killable --mem=16g -c2 --gres=gpu:1,vmem:8g --time=1-0 run_train_2d_128.sh manifold_horse_1_11
sbatch --killable --mem=16g -c2 --gres=gpu:1,vmem:8g --time=1-0 run_train_2d_128.sh manifold_kitchen_3_8


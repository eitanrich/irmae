#!/bin/bash
sbatch --killable --mem=16g -c2 --gres=gpu:1,vmem:8g --time=1-0 run_train_8d.sh manifold_bedrooms_8
sbatch --killable --mem=16g -c2 --gres=gpu:1,vmem:8g --time=1-0 run_train_8d.sh manifold_car_8
sbatch --killable --mem=16g -c2 --gres=gpu:1,vmem:8g --time=1-0 run_train_8d.sh manifold_cat_8
sbatch --killable --mem=16g -c2 --gres=gpu:1,vmem:8g --time=1-0 run_train_8d.sh manifold_church_8
sbatch --killable --mem=16g -c2 --gres=gpu:1,vmem:8g --time=1-0 run_train_8d.sh manifold_ffhq_8
sbatch --killable --mem=16g -c2 --gres=gpu:1,vmem:8g --time=1-0 run_train_8d.sh manifold_horse_8
sbatch --killable --mem=16g -c2 --gres=gpu:1,vmem:8g --time=1-0 run_train_8d.sh manifold_kitchen_8

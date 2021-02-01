#!/bin/sh
#SBATCH --job-name=GNN # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

python3 preprocessing.py --dataset_input_dir graph_gexf/REDDIT-MULTI-5K
python3 main.py --dataset_dir data_plk/REDDIT-MULTI-5K


#!/bin/bash
#SBATCH --account=def-panos
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --mail-user=aisha.eldeeb.ubc@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=output.out

root=/home/aishaeld

module load StdEnv/2020
module load gentoo/2020
module load python/3.7.7
module load intel/2020.1.217  cuda/11.4


source $root/env/bin/activate
echo "Training the Model"
python -u main.py
echo "End"
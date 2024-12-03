#!/bin/bash
#SBATCH --account=def-panos
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=72:00:00
#SBATCH --mail-user=aisha.eldeeb.ubc@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=training.out

# Load required modules
module load StdEnv/2020 cuda/11.4 cudnn/8.2.0 llvm/8 python/3.8 geos/3.8.1
export LD_LIBRARY_PATH={$LD_LIBRARY_PATH}:$CUDA_HOME/lib64:/cvmfs/soft.computecanada.ca/easybuild/software/2020/CUDA/cuda11.4/cudnn/8.2.0/lib64
export LLVM_CONFIG=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/llvm/8.0.1/bin/llvm-config
export NCCL_BLOCKING_WAIT=1


ENV_NAME="training_env"
DIR_NAME="RTFM"

# Navigate to project directory
cd /home/$USER/scratch/$DIR_NAME

# Activate virtual environment
source $ENV_NAME/bin/activate

echo "Start Training"

python main.py

echo "Training completed!"



#!/bin/bash
#SBATCH --verbose
#SBATCH --time=71:59:59
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --mail-type=ALL # select which email types will be sent
#SBATCH --mail-user=NETID@nyu.edu # NOTE: put your netid here if you want emails

#SBATCH --array=0-35 # here the number depends on number of tasks in the array, e.g. 0-11 will create 12 tasks
#SBATCH --output=./logs/%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID,
#SBATCH --error=./logs/%A_%a.err # MAKE SURE WHEN YOU RUN THIS, ../train_logs IS A VALID PATH

# #####################################################
#SBATCH --gres=gpu:1 # uncomment this line to request a gpu
#SBATCH --partition=nvidia # uncomment this line to request a gpu
#SBATCH --cpus-per-task=4

sleep $(( (RANDOM%10) + 1 )) # to avoid issues when submitting large amounts of jobs

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

echo "Job ID: ${SLURM_ARRAY_TASK_ID}"

# activate conda env
source ~/.bashrc

eval "$(conda shell.bash hook)"

conda activate soyuj

cd /scratch/sjb8193/NRC-regularizations
export PYTHONPATH=$PYTHONPATH:/scratch/sjb8193/NRC-regularizations
python main/exp/case1/carla2d_single_task.py --setting ${SLURM_ARRAY_TASK_ID}
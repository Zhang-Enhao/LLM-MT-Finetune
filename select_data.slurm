#!/bin/bash
#SBATCH --job-name=select-data
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:h100-47:1 -C cuda90
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1132290@comp.nus.edu.sg

# 打印开始信息
echo "Job $SLURM_JOB_NAME is starting..."

# 使用指定环境的Python解释器运行脚本
srun select_data.sh

# 打印结束信息
echo "Job $SLURM_JOB_NAME has finished."
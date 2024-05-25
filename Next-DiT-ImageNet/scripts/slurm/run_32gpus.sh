#!/bin/bash

# run Next-DiT with cluster

# added config here for slurm cluster using 8 GPUs

# run Next-DiT 600M
srun bash exps/600M_bs256_lr5e-4_bf16_qknorm_lognorm.sh
# run Next-DiT 2B
# srun bash exps/2B_bs256_lr5e-4_bf16_qknorm_lognorm.sh
# run Next-DiT 3B
# srun bash exps/3B_bs256_lr5e-4_bf16_qknorm_lognorm.sh
# run Next-DiT 7B
# srun bash exps/7B_bs256_lr5e-4_bf16_qknorm_lognorm.sh

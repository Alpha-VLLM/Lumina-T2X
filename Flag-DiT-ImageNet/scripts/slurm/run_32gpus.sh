#!/bin/bash

# run Flag-DiT with cluster

# added config here for slurm cluster using 32 GPUs

# run Flag-DiT 600M
srun bash exps/600M_bs256_lr5e-4_bf16_qknorm_lognorm.sh
# run Flag-DiT 3B
srun bash exps/3B_bs256_lr5e-4_bf16_qknorm_lognorm.sh
# run Flag-DiT 7B
srun bash exps/7B_bs256_lr5e-4_bf16_qknorm_lognorm.sh

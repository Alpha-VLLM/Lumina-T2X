#!/bin/bash

# run Flag-DiT with cluster

# added config here for slurm cluster using 8 GPUs

# run Flag-DiT 600M
bash exps/600M_bs256_lr5e-4_bf16_qknorm_lognorm.sh
# run Flag-DiT 3B
bash exps/3B_bs256_lr5e-4_bf16_qknorm_lognorm.sh
# run Flag-DiT 7B
bash exps/7B_bs256_lr5e-4_bf16_qknorm_lognorm.sh

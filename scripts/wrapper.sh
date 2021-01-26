#!/usr/bin/env bash

cd $SCRATCH/taskfarmer
module load python
shifter --entrypoint \
python -u /global/homes/s/sachdeva/neurobiases/scripts/cv_fit_means_task_farmer.py \
--save_path=$1 \
--model_fit=$2 \
--N=$3 \
--M=$4 \
--K=$5 \
--D=$6 \
--n_coupling_locs=$7 \
--coupling_loc_min=$8 \
--coupling_loc_max=$9 \
--coupling_loc_idx=$10 \
--n_tuning_locs=$11 \
--tuning_loc_min=$12 \
--tuning_loc_max=$13 \
--tuning_loc_idx=$14 \
--n_coupling_lambdas=$15 \
--coupling_lambda_lower=$16 \
--coupling_lambda_upper=$17 \
--n_tuning_lambdas=$18 \
--tuning_lambda_lower=$19 \
--tuning_lambda_upper=$20 \
--fine_sweep_frac=0.05 \
--max_K=1 \
--cv=5 \
--coupling_distribution=gaussian \
--coupling_sparsity=0.5 \
--coupling_scale=1. \
--tuning_distribution=gaussian \
--tuning_sparsity=0.5 \
--tuning_scale=1. \
--corr_cluster=0.25 \
--solver=cd \
--initialization=fits \
--max_iter=5000 \
--tol=1e-4 \
--refit \
--coupling_rng=1252021 \
--tuning_rng=25012021 \
--dataset_rng=2452954 \
--fitter_rng=495821
#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MAIN_DIR="${SCRIPT_DIR}/.."

LOG_DIR="${MAIN_DIR}/logs"

vae_dir="${LOG_DIR}/gulper_logs/oversample_pubchem_preprocessed_with_selfies_with_properties_small_d_bottleneck_1_d_model_256_transformer_layers_6_lr_1e-4_wd_1e-5_kl_0.2_p_1_tanimoto_shifted_euclidean_correlation_triu_0.5_batch_size_256/official_version/"
vae="${vae_dir}/checkpoints/last.ckpt"

test_name="pubchem"
data_dir="/home/hugo/data/pubchem/data/"
train="${data_dir}/pubchem_preprocessed_with_selfies_with_properties_small_train_train.csv"
test="${data_dir}/pubchem_preprocessed_with_selfies_with_properties_small_test.csv"

output_dir=${vae_dir}/"evaluation_output_on_${test_name}"
output_res_dict=${output_dir}/"evaluation_results.json"
write_reconstructed_to=${output_dir}/"reconstructed.csv"

batch_size=4096
num_workers=0
device='cuda:1'

nohup python -u ${MAIN_DIR}/chembed/evaluate.py --vae ${vae} \
												--output_res_dict ${output_res_dict} \
												--evaluate_reconstruction \
												--test_data_path ${test} \
												--write_reconstructed_to ${write_reconstructed_to} \
												--evaluate_generation \
												--nb_samples_for_generation 10000 \
												--evaluate_novelty \
												--train_data_csv ${train} \
												--batch_size ${batch_size} \
												--num_workers ${num_workers} \
												--device ${device} \
												&> "${LOG_DIR}/log_evaluate.out" &

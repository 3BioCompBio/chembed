SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MAIN_DIR="${SCRIPT_DIR}/.."

data_dir="/home/hugo/data/pubchem/data"
nb_samples=1000000
dataset_name="pubchem_preprocessed_with_selfies_with_properties_small"
prefix="${dataset_name}_train_train_with_stratum_sample_${nb_samples}"
a="1e-05"
train_name="${prefix}_train_a_${a}"
train="${data_dir}/${train_name}.parquet"
validation="${data_dir}/${prefix}_validation.parquet"
stats="${data_dir}/${dataset_name}_statistics.json"
ref="${data_dir}/${prefix}_train.parquet"

batch_size=256
max_epochs=5
learning_rate=1e-4
weight_decay=1e-5

xp_name="ablation_${train_name}_batch_${batch_size}_epochs_${max_epochs}_lr_${learning_rate}_wd_${weight_decay}"
log_dir="${MAIN_DIR}/logs/ablation_studies/${xp_name}"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python ${SCRIPT_DIR}/run_ablation_studies.py --train_path ${train} \
											--validation_path ${validation} \
											--properties_statistics_path ${stats} \
											--log_dir ${log_dir} \
											--gpu_devices 0 \
											--batch_size ${batch_size} \
											--max_epochs ${max_epochs} \
											--learning_rate ${learning_rate} \
											--weight_decay ${weight_decay} \
											--use_precomputed_fingerprints \
											--ref_file ${ref}

#export NCCL_BLOCKING_WAIT=1
#export NCCL_ASYNC_ERROR_HANDLING=1
#export NCCL_DEBUG=INFO
#export NCCL_TIMEOUT=3600
#export NCCL_DEBUG_SUBSYS=ALL

dataset_name=pubchem_preprocessed_with_selfies_with_properties_small
data_dir=/home/hugo/data/pubchem/data/

#train=$data_dir/${dataset_name}_train_train.parquet
train=$data_dir/${dataset_name}_train_train_a_1e-05.csv
validation=$data_dir/${dataset_name}_train_validation.parquet
statistics=$data_dir/${dataset_name}_statistics.json
log_dir=logs/

learning_rate=1e-4
weight_decay=1e-5
kl_div_coeff=0.2
properties_loss_coeff=1
tanimoto_loss_coeff=0.5
tanimoto_loss_type=shifted_euclidean_correlation_triu
batch_size=256
num_workers=0
max_epochs=1000000
model_class=SELFIES_Transformer_VAE
d_model=256
d_bottleneck=1
nb_layers=6
dim_feedforward_encoder=$((d_model*4))
dim_feedforward_decoder=$((d_model*2))


model_name=oversample_${dataset_name}_d_bottleneck_${d_bottleneck}_d_model_${d_model}_transformer_layers_${nb_layers}_lr_${learning_rate}_wd_${weight_decay}_kl_${kl_div_coeff}_p_${properties_loss_coeff}_tanimoto_${tanimoto_loss_type}_${tanimoto_loss_coeff}_batch_size_${batch_size}

pip install .
nohup python -u chembed/train.py --train_path ${train} \
								--validation_path ${validation} \
								--properties_statistics_path ${statistics} \
								--properties MolWt MolLogP TPSA BertzCT Kappa1 Kappa2_clipped Kappa3_clipped \
								--log_dir ${log_dir} \
								--model_name ${model_name} \
								--batch_size ${batch_size} \
								--num_workers ${num_workers} \
								--gpu_devices 0 1 \
								--max_epochs ${max_epochs} \
								--weight_decay ${weight_decay} \
								--initialize_embedding_weights \
								--learning_rate ${learning_rate} \
								--kl_div_coefficient ${kl_div_coeff} \
								--properties_loss_coefficient ${properties_loss_coeff} \
								--train_with_tanimoto_similarity \
								--tanimoto_loss_coefficient ${tanimoto_loss_coeff} \
								--tanimoto_loss_type ${tanimoto_loss_type} \
								--checkpoint_every_n_train_step 1000 \
								--checkpoint_save_top_k 1 \
								--vae_model_class ${model_class} \
								--d_bottleneck ${d_bottleneck} \
								--d_model ${d_model} \
								--nb_transformer_layers ${nb_layers} \
								--dim_feedforward_encoder ${dim_feedforward_encoder} \
								--dim_feedforward_decoder ${dim_feedforward_decoder} \
								&> logs/log_train_${model_name}.out &

#!/bin/bash

set -euo pipefail

properties_str="HeavyAtomCount MolLogP TPSA BertzCT BalabanJ Kappa1 Kappa2 Kappa3"
epsilon_quantile=0.001
properties_to_normalize=${properties_str}
properties_to_clip="Kappa2 Kappa3"
oversample_a=1e-5
test_size=0.2

num_workers=8

data_dir="/data1/hugo/data/pubchem"

CID_file="${data_dir}/CID-SMILES"

preprocessed_prefix="${data_dir}/pubchem_preprocessed"
preprocessed_file="${preprocessed_prefix}.parquet"
train="${preprocessed_prefix}_train.parquet"
test="${preprocessed_prefix}_test.parquet"
train_train="${preprocessed_prefix}_train_train.parquet"
train_validation="${preprocessed_prefix}_train_validation.parquet"
token_counts="${preprocessed_prefix}_train_token_counts.json"


stats="${preprocessed_prefix}_statistics.json"
vocab="${preprocessed_prefix}_vocab.json"


if [ ! -f ${preprocessed_file} ]; then

	# starting from a file with properties, process it
	preprocessed_props_file_processed="${preprocessed_prefix}_with_properties_processed.parquet"
	if [ ! -f ${preprocessed_props_file_processed} ]; then

		# if the file with properties doesn't exist, create one from the file with selfies
		preprocessed_props_file="${preprocessed_prefix}_with_properties.parquet"
		if [ ! -f ${preprocessed_props_file} ]; then


			# if the file with selfies doesn't exist, create it from the file with preprocessed smiles
			preprocessed_selfies_file="${preprocessed_prefix}_selfies.parquet"
			if [ ! -f ${preprocessed_selfies_file} ]; then

				# if the file with preprocessed smiles doesn't exist, create it
				preprocessed_smiles_file="${preprocessed_prefix}_smiles_only.parquet"
				if [ ! -f ${preprocessed_smiles_file} ]; then
					python -u scripts/preprocess_pubchem_smiles.py  --input_file ${CID_file} \
																	--output_file ${preprocessed_smiles_file} \
																	--num_workers ${num_workers}
				fi


				# try to encode to SELFIES, drop rows that didn't work
				python -u scripts/add_selfies.py --input_file ${preprocessed_smiles_file} \
													--output_file ${preprocessed_selfies_file} \
													--num_workers ${num_workers} \
													--already_standardized \
													--drop_fail_encode
				rm -f ${preprocessed_smiles_file}
			fi


			python -u scripts/add_properties.py --input_file ${preprocessed_selfies_file} \
												--output_file ${preprocessed_props_file} \
												--num_workers ${num_workers} \
												--properties ${properties_str}
			rm -f ${preprocessed_selfies_file}
		fi


		python -u scripts/preprocess_properties.py ${preprocessed_props_file} \
												${preprocessed_props_file_processed} \
												--properties_to_clip ${properties_to_clip} \
												--epsilon_quantile ${epsilon_quantile} \
												--properties_to_normalize ${properties_to_normalize} \
												--output_stats ${stats}
		rm -f ${preprocessed_props_file}
	fi

	# copy file to final file
	cp ${preprocessed_props_file_processed} ${preprocessed_file}
	rm -f ${preprocessed_props_file_processed}

fi


# build vocab
if [ ! -f ${vocab} ]; then
	python -u scripts/build_vocab.py ${preprocessed_file} \
									${vocab} \
									--num_workers ${num_workers}
fi

# split train/test
if [ ! -f ${train} ]; then
	python -u scripts/split_random_train_test.py ${preprocessed_file} \
												--random_seed 1 \
												--test_size ${test_size} \
												--output_train ${train} \
												--output_test ${test}
fi


# split train/validation
if [ ! -f ${train_train} ]; then
	python -u scripts/split_random_train_test.py ${train} \
												--random_seed 1 \
												--test_size ${test_size} \
												--output_train ${train_train} \
												--output_test ${train_validation}

fi


# compute token counts
if [ ! -f ${token_counts} ]; then
	python -u scripts/get_overall_token_counts.py ${train} \
												${token_counts}
fi


# oversample train
python -u scripts/duplicate_samples_given_token_frequencies.py ${train_train} \
															--a ${oversample_a} \
															--token_counts ${token_counts}

#!/bin/bash

set -e

Tartarus_dir=$1
chembed_dir=$2

containers_dir=${GLOBALSCRATCH}/containers
container_name=tartarus_xp_apptainer_with_chembed
sandbox_dir=${containers_dir}/${container_name}.sandbox
sif_file=${containers_dir}/${container_name}.sif

# build sandbox from miniconda3 apptainer
original_sif=${containers_dir}/miniconda3_latest.sif
apptainer pull ${original_sif} docker://continuumio/miniconda3
apptainer build --sandbox ${sandbox_dir} ${original_sif}
rm ${original_sif}

# create conda env from environment file
apptainer shell --bind ${containers_dir}:/mnt/ --writable ${sandbox_dir} << "EOF"
conda env update --file /mnt/tartarus_env.yml --prune
EOF

# configure conda env for xtb
apptainer exec --bind ${tmp_dir}:/tmp --writable ${sandbox_dir} bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate tartarus_singularity_env && echo 'export XTBHOME=\$CONDA_PREFIX' >> \$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh && echo 'source \"\$CONDA_PREFIX/share/xtb/config_env.bash\"' >> \"\$CONDA_PREFIX/etc/conda/activate.d/xtb.sh\" && echo 'export LD_PRELOAD=\$LD_PRELOAD:\$CONDA_PREFIX/lib/libgomp.so.1.0.0' >> \$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh && echo 'export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH' >> \$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh "

# install additional pip dependencies
apptainer exec --bind ${tmp_dir}:/tmp --writable ${sandbox_dir} bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate tartarus_singularity_env && pip install geodesic-interpolate --extra-index-url https://test.pypi.org/simple/"

# install Tartarus in conda env
apptainer exec --bind ${Tartarus_dir}:/mnt --writable ${sandbox_dir} bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate tartarus_singularity_env && cd /mnt/ && pip install ."

# install chembed in conda env
apptainer exec --bind ${chembed_dir}:/mnt --bind ${tmp_dir}:/tmp --writable ${sandbox_dir} bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate tartarus_singularity_env && cd /mnt/ && pip install ."

# export sandbox to sif file
apptainer build ${sif_file} ${sandbox_dir}

#!/bin/bash
#Set job requirements
#SBATCH -n 1
#SBATCH -t 5:00
#SBATCH -p 'gpu'
#SBATCH --gpus-per-node=1
 
#Loading modules
module load 2022
module load SciPy-bundle/2022.05-foss-2022a

#Copy input file to scratch
cd $HOME/GOGGLE/
 
#Create output directory on scratch
mkdir "$TMPDIR"/output_dir

conda env create -f goggle.yml
source activate goggle_env

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
python $HOME/GOGGLE/setup.py build 
python $HOME/GOGGLE/setup.py install 
python $HOME/GOGGLE/src/goggle/exps/synthetic_data/exp_red_wine.py 
 
#Copy output directory from scratch to home
cp -r "$TMPDIR"/output_dir $HOME

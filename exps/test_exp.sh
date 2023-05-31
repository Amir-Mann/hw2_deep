#!/bin/bash

###
# CS236781: Deep Learning
# py-sbatch.sh
#
# This script runs python from within our conda env as a slurm batch job.
# All arguments passed to this script are passed directly to the python
# interpreter.
#

###
# Example usage:
#
# Running the prepare-submission command from main.py as a batch job
# ./py-sbatch.sh main.py prepare-submission --id 123456789
#
# Running all notebooks without preparing a submission
# ./py-sbatch.sh main.py run-nb *.ipynb
#
# Running any other python script myscript.py with arguments
# ./py-sbatch.sh myscript.py --arg1 --arg2=val2
#

###
# Parameters for sbatch
#
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
JOB_NAME="test_hp"
MAIL_USER="amir.mann@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=cs236781-hw

sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	-o 'slurm-%N-%j.out' \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/fprofile.d/conda.sh
conda activate $CONDA_ENV

# Run python with the args to the script
python -m hw2.experiments run-exp -s 20773 -H 90 -P 1 --epochs 30 --reg 0.0001 --lr 0.0005 -n test_L2_K32_lr0.0001 -K 32 -L 2
python -m hw2.experiments run-exp -s 20773 -H 90 -P 1 --epochs 30 --reg 0.0005 --lr 0.0005 -n test_L2_K32_lr0.0005 -K 32 -L 2
python -m hw2.experiments run-exp -s 20773 -H 90 -P 1 --epochs 30 --reg 0.001  --lr 0.0005 -n test_L2_K32_lr0.001  -K 32 -L 2
python -m hw2.experiments run-exp -s 20773 -H 90 -P 1 --epochs 30 --reg 0.005  --lr 0.0005 -n test_L2_K32_lr0.005  -K 32 -L 2
python -m hw2.experiments run-exp -s 20773 -H 90 -P 1 --epochs 30 --reg 0.01   --lr 0.0005 -n test_L2_K32_lr0.01   -K 32 -L 2
python -m hw2.experiments run-exp -s 20773 -H 90 -P 1 --epochs 30 --reg 0.05   --lr 0.0005 -n test_L2_K32_lr0.05   -K 32 -L 2

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF


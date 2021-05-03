#!/bin/bash
#SBATCH --partition=long                      # Ask for unkillable job
#SBATCH --cpus-per-task=4                     # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=20G                             # Ask for 10 GB of RAM
#SBATCH --time=1:00:00                        # The job will run for 3 hours
#SBATCH -o /miniscratch/darshan.patil/slurm/slurm-%j.out  # Write the log on tmp1


MODEL="tem"
REPLAY_WEIGHT="1.0"
HOPFIELD_PROB=".5"
PROJECT_NAME="cont_hopfield_final_2"
RUN_NAME="${MODEL}_hp${HOPFIELD_PROB}"
# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate hopfield

# 3. Copy your dataset on the compute node
mkdir $SLURM_TMPDIR/data
cp -r /network/projects/d/darshan.patil/data/cifar100 $SLURM_TMPDIR/data/

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR

for SEED in {1..5}
do
    OUTPUT_FILE="run_${SEED}.json"
    python main.py -m ${MODEL}  \
                    --data-dir $SLURM_TMPDIR/data/cifar100  \
                    --replay-weight ${REPLAY_WEIGHT} \
                    --hopfield-prob ${HOPFIELD_PROB} \
                    --project-name ${PROJECT_NAME} \
                    --run-name ${RUN_NAME}\
                    --output-file ${OUTPUT_FILE} \
                    --seed ${SEED} 
done
# 5. Copy whatever you want to save on $SCRATCH
# cp $SLURM_TMPDIR/<to_save> /network/tmp1/<user>/
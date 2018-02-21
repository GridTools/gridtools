#!/bin/bash
#SBATCH --nodes=<MPI_NODES>
#SBATCH --ntasks=<MPI_TASKS>
#SBATCH --ntasks-per-node=<MPI_PPN>
#SBATCH --output=<OUTPUTFILE>
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=<CPUSPERTASK>
#SBATCH --account=c14
#SBATCH --partition=<QUEUE>

echo $LD_LIBRARY_PATH

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# set environmental parameters

<JOB_ENV>

<CMD>


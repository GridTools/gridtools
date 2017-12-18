#!/bin/bash
#SBATCH --job-name=gridtools_test
#SBATCH --nodes=<MPI_NODES>
#SBATCH --ntasks=<MPI_TASKS>
#SBATCH --ntasks-per-node=<MPI_PPN>
#SBATCH --output=<OUTPUTFILE>
#SBATCH --partition=<QUEUE>
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:<MPI_TASKS>
#SBATCH --cpus-per-task=<CPUSPERTASK>

echo $LD_LIBRARY_PATH

<JOB_ENV>

echo "RUNNING IN NODE ${HOSTNAME}"

<CMD>

sync

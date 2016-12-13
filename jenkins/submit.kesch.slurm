#!/bin/bash
#SBATCH --job-name=cosmo-e
#SBATCH --nodes=<MPI_NODES>
#SBATCH --ntasks=<MPI_TASKS>
#SBATCH --ntasks-per-node=<MPI_PPN>
#SBATCH --output=<OUTPUTFILE>
#SBATCH --partition=<QUEUE>
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:<MPI_TASKS>
####SBATCH --distribution block:block
#SBATCH --cpus-per-task=<CPUSPERTASK>

echo $LD_LIBRARY_PATH

# set environmental parameters
# export OMP_NUM_THREADS=<CPUSPERTASK>
# if [[ <CPUSPERTASK> > 1 ]]; then
#     last=$((<CPUSPERTASK>-1))
#     export GOMP_CPU_AFFINITY=`seq 0 $last`
# fi

# export MALLOC_MMAP_MAX_=0
# export MALLOC_TRIM_THRESHOLD_=536870912
# export COSMO_NPROC_NODEVICE=0

<JOB_ENV>

echo "RUNNING IN NODE ${HOSTNAME}"

<CMD>

sync
#!/bin/bash
#SBATCH --job-name=cosmo-e
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --output=test.out
#SBATCH --partition=<QUEUE>
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --distribution block:block
#SBATCH --cpus-per-task=<CPUSPERTASK>

echo $LD_LIBRARY_PATH

# set environmental parameters
export OMP_NUM_THREADS=<CPUSPERTASK>
if [[ <CPUSPERTASK> > 1 ]]; then
    last=$((<CPUSPERTASK>-1))
    export GOMP_CPU_AFFINITY=`seq 0 $last`
fi

export MALLOC_MMAP_MAX_=0
export MALLOC_TRIM_THRESHOLD_=536870912
export COSMO_NPROC_NODEVICE=0
export CUDA_AUTO_BOOST=0

export GCLOCK=875
export G2G=1

echo "RUNNING IN NODE ${HOSTNAME}"

<CMD>


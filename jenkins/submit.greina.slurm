#!/bin/bash
#SBATCH --job-name=gridtools.test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --output=test.out
#SBATCH --partition=<QUEUE>
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=<CPUSPERTASK>

echo $LD_LIBRARY_PATH

# set environmental parameters
export OMP_NUM_THREADS=<CPUSPERTASK>
if [[ <CPUSPERTASK> ]]; then
    last=$((<CPUSPERTASK>-1))
    export GOMP_CPU_AFFINITY=`seq 0 $last`
fi

# set environmental parameters
export OMP_NUM_THREADS=1
export MALLOC_MMAP_MAX_=0
export MALLOC_TRIM_THRESHOLD_=536870912
export CUDA_AUTO_BOOST=0

export GCLOCK=875
export G2G=1

echo "RUNNING IN NODE ${HOSTNAME}"

<CMD>


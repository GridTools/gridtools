#!/bin/bash

prefix=/home/kardoj/gridtools/build/build
tasks_per_node=4

max_it=50
nsamples=5
nthreads=12
eps=1e-14

for n in 128 ; do
for nproc in 1 2 4 8 16 32 64 128 256; do

comp_nodes=$((nproc/tasks_per_node))
if [ ${comp_nodes} -eq 0 ]
then
    comp_nodes=1
fi

qsub <<-_EOF
#!/bin/bash -l
#
#PBS -N gridtools
#PBS -A DD-16-7
#PBS -l select=${comp_nodes}
#PBS -l walltime=06:00:00
#PBS -e cg_${n}_${comp_nodes}_${nproc}.e
#PBS -o cg_${n}_${comp_nodes}_${nproc}.o

# Load modules
. ~/gridtools/gridtools_setup.sh

# cd to the directory from where the job was started
cd $PBS_O_WORKDIR

# Run job
OMP_NUM_THREADS=${nthreads} mpirun -n ${nproc} -ppn ${tasks_per_node} ${prefix}/cg_naive_block $n $n $n $max_it $eps $nsamples
_EOF

done
done

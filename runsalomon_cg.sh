#!/bin/bash

nt=500
nthreads=16

#atol=1e-50
#rtol=1e-5

eps=1e-8

for n in 64 ; do
for comp_nodes in 1 2 4; do

qsub <<-_EOF
#!/bin/bash -l
#
#PBS -N gridtools
#PBS -A DD-16-7
#PBS -l select=${comp_nodes}
#PBS -l walltime=01:00:00
#PBS -e cg_${n}_${comp_nodes}.e
#PBS -o cg_${n}_${comp_nodes}.o

# Load modules
module load imkl/2017.0.098-iimpi-2017.00-GCC-5.4.0-2.26

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kardoj/lib/pardiso
export PARDISOLICMESSAGE=1

# cd to the directory from where the job was started
cd $PBS_O_WORKDIR

# Run job
OMP_NUM_THREADS=${nthreads} mpirun -np ${comp_nodes} -perhost 1 /home/kardoj/gridtools/build/build/cg_naive_block $n $n $n $nt $eps
_EOF

done
done

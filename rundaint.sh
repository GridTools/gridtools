#!/bin/bash

nt=500
nthreads=8

#atol=1e-50
#rtol=1e-5

eps=1e-8

for n in 64 96 128 256 512 ; do
#for comp_nodes in 1 2 4 8 16 32 64 128  ; do
for comp_nodes in 32 64 128  ; do

sbatch <<-_EOF
#!/bin/bash
#SBATCH --account=u3
#SBATCH --job-name=naive_${comp_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=${comp_nodes}
#SBATCH --time=01:00:00
#SBATCH --output=cg_${n}_${comp_nodes}.o

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/jkardos/PowerGrid
export PARDISOLICMESSAGE=1

OMP_NUM_THREADS=${nthreads} srun -c ${nthreads} -n ${comp_nodes} --ntasks-per-node 1 --hint=nomultithread /users/jkardos/gridtools/build/build/cg_naive_block $n $n $n $nt $eps
_EOF

done
done

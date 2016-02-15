#!/bin/bash -l

#SBATCH --job-name="GridTools"
#--SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --output=output.%j.o
#SBATCH --error=output.%j.o

nt=2

#======START=====
for i in 32 64 128
do
    echo "Grid size $i x $i x $i, $nt steps"
    aprun /users/jkardos/gridtools/build_optim/build/cg_block $i $i $i $nt
done
#=====END====

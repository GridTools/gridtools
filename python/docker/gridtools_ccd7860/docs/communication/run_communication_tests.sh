#!/bin/bash

module ()
{
    eval `/apps/dom/Modules/3.2.10/bin/modulecmd bash $*`
}
module load boost
module load gcc/4.8.2 
module load cuda/6.5
module load mvapich2/1.9-gcc-4.8.2
SUCCESS=true
export MV2_USE_CUDA=1
NP=$[$RANDOM % 8 + 1]
if [ "$USE_GPU" == "ON" ]
then
for name in cuda_generic cuda_all_test cuda_all_test_2 ; do 
    for t in 0 1 2 3 4 5 6 7 ; do 
        echo $name $t ; 
        rm out* 2>/dev/null ; 
        mpiexec.hydra -np $NP ./build/test_halo_exchange_3D_$name\_$t 10 12 13  1 2 1  1 2 2 1 1 1  >/dev/null || SUCCESS=false
        ~mbianco/bin/count2 "out*" RES PAS FAI || SUCCESS=false; 
    done; 
done
else
for name in generic all_test all_test_2 ; do 
    for t in 0 1 2 3 4 5 6 7 ; do 
        echo $name $t ; 
        rm out* 2>/dev/null ; 
        mpiexec.hydra -np $NP ./build/test_halo_exchange_3D_$name\_$t 10 12 13  1 2 1  1 2 2 1 1 1  >/dev/null || SUCCESS=false
        ~mbianco/bin/count2 "out*" RES PAS FAI || SUCCESS=false; 
    done; 
done
fi

`$SUCCESS`

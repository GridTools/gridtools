#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Run as ./processResults.sh DIR, where DIR is directory with the files to process."
    exit
fi

prefix=$1

#phases of the solution process
phase=(TIMER_GLOBAL TIMER_COMPUTE_STENCIL_INNER TIMER_COMPUTE_DOTPROD TIMER_HALO_PACK TIMER_HALO_ISEND_IRECV TIMER_HALO_UNPACK_WAIT)

#domain sizes
domains=(128 256 512 1024)

#create header of the data table
printf "Id Domain Nodes Partitions "
for match in "${phase[@]}"
do
    printf "${match} "
done
printf "\n"


for d in "${domains[@]}"
do

    id=0

    for f in `ls -v ${prefix}/cg_${d}*.o`
    do

        #Match N NN NP  in cg_N_NN_NP.o filename
        #N domain size
        #NN number of nodes
        #NP number of partitions
        N=`echo $f | sed -e 's/.*cg_\([0-9]*\)_\([0-9]*\)_\([0-9]*\).*o/\1/g'`
        NN=`echo $f | sed -e 's/.*cg_\([0-9]*\)_\([0-9]*\)_\([0-9]*\).*o/\2/g'`
        NP=`echo $f | sed -e 's/.*cg_\([0-9]*\)_\([0-9]*\)_\([0-9]*\).*o/\3/g'`
        printf "${id} ${N} ${NN} ${NP} "

        for match in "${phase[@]}"
        do
            T=`cat $f | grep "${match} "  | sed -e "s/${match}[ ]*=[ ]*\([0-9].[0-9]*e[+-][0-9]*\) s/\1/g"`
            printf "${T} "
        done
        printf "\n"

        id=$((id + 1))

    done
    printf "\n"

done


#Id Domain Partitions TIMER_GLOBAL TIMER_COMPUTE_STENCIL_BORDER TIMER_COMPUTE_DOTPROD TIMER_COMPUTE_RNG TIMER_COMPUTE_MISC TIMER_HALO_PACK TIMER_HALO_ISEND_IRECV TIMER_HALO_UNPACK_WAIT TIMER_COMM_DOTPROD
#0 256 1 4.7622e+01 3.2074e+00 1.0459e+01 9.9293e-01 2.5943e+01 0.0000e+00 4.8637e-05 4.2677e-05 2.8462e-03
#1 256 2 2.5872e+01 3.0698e+00 5.3315e+00 4.9654e-01 1.3097e+01 1.6893e-01 1.3096e-03 2.3200e-01 8.3187e-03
#2 256 4 1.3388e+01 1.5856e+00 2.7270e+00 2.4919e-01 6.7084e+00 9.7960e-02 1.5702e-03 2.1627e-01 1.4656e-02
#3 256 8 8.3075e+00 8.3120e-01 1.6158e+00 1.2533e-01 4.3305e+00 6.4987e-02 2.3093e-03 1.7128e-01 3.5113e-02
#4 256 16 7.3619e+00 8.4301e-01 1.3299e+00 6.3370e-02 3.8645e+00 7.2764e-02 2.8334e-03 1.2281e-01 8.3157e-02
#5 256 32 8.8839e+00 5.6850e-01 7.9281e-01 5.5293e-02 2.9143e+00 4.7666e-02 6.0103e-03 6.9118e-01 2.8379e+00
#6 256 64 7.6713e+00 2.4418e-01 1.4308e+00 3.2449e-02 1.5541e+00 2.8387e-02 1.2045e-02 1.0925e+00 2.9358e+00
#7 256 128 7.3206e+00 1.7126e-01 1.0231e-01 8.4929e-03 5.4789e-01 1.4975e-02 9.4099e-03 1.4967e+00 4.8413e+00


FAIL=false
for gpu in ON OFF; do
    for mpi in ON OFF; do
        for build in DEBUG RELEASE; do
            for cpp11 in ON OFF; do
                name=TC_OUT_$gpu\_$mpi\_$build\_$cpp11.txt
                if [ "$FAIL" == "false" ]; then
                    echo FILE NAME BEING PRODUCED $name
                    cmake \
                        -DCUDA_NVCC_FLAGS:STRING="-arch=sm_35" \
                        -DUSE_GPU:BOOL=$gpu \
                        -DUSE_MPI:BOOL=$mpi \
                        -DUSE_MPI_COMPILER:BOOL=OFF \
                        -DCMAKE_BUILD_TYPE:STRING=$build\
                        -DCMAKE_CXX_FLAGS:STRING=" -fopenmp -O3 -m64 -mavx -DNDEBUG -DBOOST_SYSTEM_NO_DEPRECATED"  \
                        -DSINGLE_PRECISION=$precision \
                        -DENABLE_CXX11=$cpp11 \
                        .

                    time make -j8 2>$name || FAIL=true
                    if [ "$FAIL" == "true" ]; then
                        echo "*************************************************************"
                        echo "*************************************************************"
                        echo "*************************************************************"
                        grep -i error $name
                        echo "*************************************************************"
                        echo "*************************************************************"
                        echo "************************************************************"
                    else
                        echo
                        echo
                        echo EVERYTTHING SEEMS FINE
                        echo
                        echo
                    fi
                fi
            done
        done
    done
done


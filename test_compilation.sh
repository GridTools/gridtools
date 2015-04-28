
# Launch the script when you have already a CMake environment set
# The script will change only those variables that are involved in the
# testing. If something is wrong it will stop and by running "make"
# you can see what compilation error you are having. You could also
# look at the generated files, but they are mangled by the -j option.

FAIL=false
for gpu in ON OFF; do
    for mpi in ON OFF; do
        for build in DEBUG RELEASE; do
            for cpp11 in ON OFF; do
                name=TC_OUT_$gpu\_$mpi\_$build\_$cpp11.txt
                if [ "$FAIL" == "false" ]; then
                    echo FILE NAME BEING PRODUCED $name
                    cmake \
                        -DUSE_GPU:BOOL=$gpu \
                        -DUSE_MPI:BOOL=$mpi \
                        -DCMAKE_BUILD_TYPE:STRING=$build\
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
                        rm $name
                    fi
                fi
            done
        done
    done
done


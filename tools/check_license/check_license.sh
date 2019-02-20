#!/bin/bash
# run this script from the root directory
if [ ! -f $PWD/LICENSE_HEADER ] ; then
    echo "you have to run the script ./tools/check_license/check_license.sh from the root of the sources"
    exit -1
fi

execludelist=(\
    "./unit_tests/interface/repository/repository_double.h" \
    "./unit_tests/interface/repository/repository_float.h" \
    "./regression/c_bindings/implementation_float.h" \
    "./regression/c_bindings/implementation_double.h" \
    "./regression/c_bindings/implementation_wrapper_double.h" \ 
    "./regression/c_bindings/implementation_wrapper_float.h" \
    )

for file in `find . -regextype posix-egrep -regex ".*\.(h|hpp|cpp|cu)$" -not -path "./build*" -not -path "./tools/*" -not -path "./docs/*"`; do

    if [[ ! " ${execludelist[@]} " =~ " ${file} " ]]; then # if not in exclude list
        if ! grep -q "Copyright (c) " $file ; then
            echo $file
            if [ "$1" == "fix" ]; then
                echo "$(cat LICENSE_HEADER)
$(cat $file)" > $file
            fi
            else # if the license matches the string above
            if [ "$1" == "update" ]; then
                sed -i '0,/\*\//d' $file
                echo "$(cat LICENSE_HEADER)
$(cat $file)" > $file
            fi
        fi
    fi
done

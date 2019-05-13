#!/bin/bash

# use the machines python virtualenv with required modules installed
source /project/c14/jenkins/python-venvs/${label%%-*}/bin/activate

grid=structured

for domain in 128 256; do
    for real_type in float double; do
        for label in daint kesch tave; do
            resultdirs=$(ls -d /project/c14/jenkins/gridtools-performance-history-new/$grid/$real_type/$domain/${label}_*)
            results=''
            for backend in cuda mc x86; do
                for dir in $resultdirs; do
                    results="$results $(ls -t $dir/*.$backend.json 2> /dev/null | head -n 1)"
                done
            done

            if [[ -n "$results" ]]; then
                ./pyutils/driver.py -v perftest plot compare -i $results -o compare-$label-$real_type-$domain.png
            fi
        done
    done
done

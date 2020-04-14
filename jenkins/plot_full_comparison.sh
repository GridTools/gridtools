#!/bin/bash

# use the machines python virtualenv with required modules installed
source /project/c14/jenkins/python-venvs/${label%%-*}/bin/activate


for domain in 128 256; do
    for real_type in float double; do
        for label in daint; do
            resultdirs=$(ls -d /project/c14/jenkins/gridtools-performance-history-gt2/${label}_*/$domain)
            results=''
            for dir in $resultdirs; do
                results="$results $(ls -t $dir/*.json 2> /dev/null | head -n 1)"
            done

            if [[ -n "$results" ]]; then
                ./pyutils/driver.py -v perftest plot compare-backends -i $results -o backends-$label-$domain.html
            fi
        done
    done
done

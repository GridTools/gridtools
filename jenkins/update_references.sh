#!/bin/bash

if [[ -z $1 ]]; then
    echo "Usage: $0 <Jenkins build number>"
    exit 1
fi

refpath=$(dirname "$BASH_SOURCE")/../pyutils/perftest/references

grid=structured
for real_type in float double; do
    for domain in 128 256; do
        for label in tave kesch daint-cn; do
            for env in gcc clang nvcc_gcc nvcc_clang clang_nvcc; do
                for backend in x86 mc cuda; do
                    current="$grid/$real_type/$domain/${label%-*}_$env/result.$backend.json"
                    src="http://jenkins-mch.cscs.ch/view/GridTools/job/GridTools_perftest_PR/$1/env=$env,label=$label,real_type=$real_type/artifact/build/pyutils/perftest/results/$current"

                    tmp=$(mktemp)
                    wget -q -O "$tmp" "$src"

                    if [[ $? == 0 ]]; then
                        dst="$refpath/$current"
                        mkdir -p "$(dirname "$dst")"
                        mv "$tmp" "$dst"
                        echo "Updated $current"
                    else
                        rm "$tmp"
                    fi
                done
            done
        done
    done
done

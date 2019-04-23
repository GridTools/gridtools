#!/bin/bash

if [[ -z $1 ]]; then
    echo "Usage: $0 <Jenkins build number>"
    exit 1
fi

refpath=$(dirname "$BASH_SOURCE")/../pyutils/perftest/references/

for f in $(find $refpath -name 'result.*.json' -print); do
    f=${f#$refpath}
    echo $f
    read grid real_type domain label env file <<< $(echo $f | sed 's#/# #g' | sed 's#_# #')

    if [[ $label == "daint" ]]; then
        label="daint-cn"
    fi

    src="http://jenkins-mch.cscs.ch/view/GridTools/job/GridTools_perftest_PR_dev2/7/env=$env,label=$label,real_type=$real_type/artifact/build/pyutils/perftest/results/$f"

    echo "Dowloading $src"
    tmp=$(mktemp)
    wget "$src" -O "$tmp"

    if [[ $err == 0 ]]; then
        mv "$tmp" "$refpath$f"
        echo "Updated $f"
    else
        echo "Skipped $f"
    fi
done


#!/bin/bash

source $(dirname "$0")/setup.sh

grid=structured

export GTCMAKE_GT_ENABLE_BACKEND_NAIVE=OFF

# build binaries for performance tests
./pyutils/builddriver.py -vvv -b release -p $real_type -g $grid -o build -d $target -c $compiler -t perftests || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }

for domain in 128 256; do
  # result directory, create if it does not exist yet
  resultdir=./build/pyutils/perftest/results/$grid/$real_type/$domain/${label}_${target}_$compiler
  mkdir -p $resultdir

  # run performance tests
  ./build/pyutils/perfdriver.py -vvv run -s $domain $domain 80 -o $resultdir/result.json || { echo 'Running failed'; rm -rf $tmpdir; exit 1; }

  allresults=''
  for backend in cuda x86 mc; do
    result=$resultdir/result.$backend.json
    if [[ -f $result ]]; then
      # append result
      allresults="$allresults $result"

      # find references for same configuration
      reference_path=./pyutils/perftest/references/$grid/$real_type/$domain/${label}_${target}_$compiler
      if [[ -d $reference_path ]]; then
        stella_reference=$(find $reference_path -name stella.json)
        gridtools_reference=$(find $reference_path -name "result.$backend.json")
        references="$stella_reference $gridtools_reference"
      else
        references=""
      fi

      # plot comparison of current result with references
      ./build/pyutils/perfdriver.py -vvv plot compare -i $references $resultdir/result.$backend.json -o plot-$backend-$domain.png || { echo 'Plotting failed'; rm -rf $tmpdir; exit 1; }
    fi
  done

  if [[ -n "$allresults" ]]; then
    # plot comparison of backends
    ./build/pyutils/perfdriver.py -vvv plot compare -i $allresults -o plot-backend-compare.png || { echo 'Plotting failed'; rm -rf $tmpdir; exit 1; }
  fi
done

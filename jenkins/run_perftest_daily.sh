#!/bin/bash

source $(dirname "$0")/setup.sh

grid=structured

export GTCMAKE_GT_ENABLE_BACKEND_NAIVE=OFF

# build binaries for performance tests
./pyutils/builddriver.py -vvv -b release -p $real_type -g $grid -o build -e $envfile -t perftests || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }

for domain in 128 256; do
  # result directory, create if it does not exist yet
  resultdir=/project/c14/jenkins/gridtools-performance-history-new/$grid/$real_type/$domain/${label}_$env
  mkdir -p $resultdir

  # name result file by date/time
  resultname=$(date +%F-%H-%M-%S).json

  # run performance tests
  ./build/pyutils/perfdriver.py -vvv run -s $domain $domain 80 -o $resultdir/$resultname || { echo 'Running failed'; rm -rf $tmpdir; exit 1; }

  for backend in cuda x86 mc; do
    # find previous results for history plot
    results=$(find $resultdir -regex ".*\.$backend\.json")
    if [[ -n "$results" ]]; then
      # plot history
      ./build/pyutils/perfdriver.py -vvv plot history -i $results -o history-$backend-$domain-full.png || { echo 'Plotting failed'; rm -rf $tmpdir; exit 1; }
      ./build/pyutils/perfdriver.py -vvv plot history -i $results -o history-$backend-$domain-last.png --limit=10 || { echo 'Plotting failed'; rm -rf $tmpdir; exit 1; }
    fi
  done
done

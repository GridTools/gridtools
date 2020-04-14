#!/bin/bash

source $(dirname "$0")/setup.sh

# build binaries for performance tests
./pyutils/driver.py -v -l $logfile build -b release -o build -e $envfile -t perftests || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }

for domain in 128 256; do
  # result directory, create if it does not exist yet
  resultdir=/project/c14/jenkins/gridtools-performance-history-gt2/${label}_$env/$domain
  mkdir -p $resultdir

  # name result file by date/time
  result="$resultdir/$(date +%F-%H-%M-%S).json"

  # run performance tests
  ./build/pyutils/driver.py -v -l $logfile perftest run -s $domain $domain 80 -o $result || { echo 'Running failed'; rm -rf $tmpdir; exit 1; }

  # find previous results for history plot
  results=$(find $resultdir -name '*.json')
  if [[ -n "$results" ]]; then
    # plot history
    ./build/pyutils/driver.py -v -l $logfile perftest plot history -i $results -o history-$domain-full.html || { echo 'Plotting failed'; rm -rf $tmpdir; exit 1; }
    ./build/pyutils/driver.py -v -l $logfile perftest plot history -i $results -o history-$domain-last.html --limit=10 || { echo 'Plotting failed'; rm -rf $tmpdir; exit 1; }
  fi

  # plot backend comparison
  ./build/pyutils/driver.py -v -l $logfile perftest plot compare-backends -i $result -o backends-$domain.html || { echo 'Plotting failed'; rm -rf $tmpdir; exit 1; }
done

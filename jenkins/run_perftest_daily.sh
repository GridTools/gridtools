# remove -cn from label (for daint)
label=${label%%-*}

# use the machines python virtualenv with required modules (matplotlib) installed
source /project/c14/jenkins/python-venvs/$label/bin/activate

if [[ $label != "kesch" ]]; then
    export SLURM_ACCOUNT=c14
    export SBATCH_ACCOUNT=c14
fi

# create directory for temporaries
if [[ $label == "tave" ]]; then
    # use /dev/shm on Tave due to small /tmp size
    tmpdir=$(mktemp -d /dev/shm/gridtools-tmp-XXXXXXXXXX)
else
    # use a subdirectory of /tmp on other systems to avoid memory problems
    tmpdir=$(mktemp -d /tmp/gridtools-tmp-XXXXXXXXXX)
fi
mkdir -p $tmpdir
export TMPDIR=$tmpdir

grid=structured

export GTCMAKE_GT_ENABLE_BACKEND_NAIVE=OFF

# build binaries for performance tests
./pyutils/builddriver.py -vvv -b release -p $real_type -g $grid -o build -d $target -c $compiler -t perftests || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }

for domain in 128 256; do
  # result directory, create if it does not exist yet
  resultdir=/project/c14/jenkins/gridtools-performance-history-new/$grid/$real_type/$domain/${label}_${target}_$compiler
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

# clean possible temporary leftovers
rm -rf $tmpdir

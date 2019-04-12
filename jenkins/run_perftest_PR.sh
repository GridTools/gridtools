# use the machines python virtualenv with required modules (matplotlib) installed
source /project/c14/jenkins/python-venvs/${label%%-*}/bin/activate

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

# build binaries for performance tests
./pyutils/builddriver.py -vvv -b release -p $real_type -g $grid -o build -d $target -c $compiler -t perftests || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }

for domain in 128 256; do
  # result directory, create if it does not exist yet
  resultdir=./build/pyutils/perftest/results/$grid/$real_type/$domain/${label%%-*}_${target}_$compiler
  mkdir -p $resultdir
  
  # run performance tests
  ./build/pyutils/perfdriver.py -vvv run -s $domain $domain 80 -o $resultdir/result.json || { echo 'Running failed'; rm -rf $tmpdir; exit 1; }
  
  # find references for same config-grid-real_type-backend combination
  reference_path=./pyutils/perftest/references/$grid/$real_type/$domain/${label%%-*}_${target}_$compiler
  if [ -d $reference_path ]; then
  	stella_reference=$(find $reference_path -name stella.json)
  	gridtools_references=$(find $reference_path -name 'result.*.json')
  	references="$stella_reference $gridtools_references"
  else
  	references=""
  fi
  
  # plot comparison of current result with references
  ./build/pyutils/perfdriver.py -vvv plot compare -i $references $resultdir/result.*.json -o plot-$domain.png || { echo 'Plotting failed'; rm -rf $tmpdir; exit 1; }
done

# clean possible temporary leftovers
rm -rf $tmpdir

# use the machines python virtualenv with required modules (matplotlib) installed
source /project/c14/jenkins/python-venvs/${label%%-*}/bin/activate

# get config name, consisting of label without postfix (like the -cn from daint-cn) and compiler name
config=${label%%-*}_$compiler

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

# build binaries for performance tests
./pyutils/build.py -vvv --build-type release --backend $backend --precision $precision --grid $grid --build-dir build --config $config --perftest-targets || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }

for domain in 128 256; do
  # result directory, create if it does not exist yet
  resultdir=./build/pyutils/results/$config/$grid/$precision/$backend/$domain
  mkdir -p $resultdir
  
  # run performance tests
  ./build/pyutils/perfdriver.py -vvv run -d $domain $domain 80 --config $config -o $resultdir/result.json || { echo 'Running failed'; rm -rf $tmpdir; exit 1; }
  
  # find references for same config-grid-precision-backend combination
  reference_path=./pyutils/references/$config/$grid/$precision/$backend/$domain
  if [ -d $reference_path ]; then
  	stella_reference=$(find $reference_path -name stella.json)
  	gridtools_reference=$(find $reference_path -name result.json)
  	references="$stella_reference $gridtools_reference"
  else
  	references=""
  fi
  
  # plot comparison of current result with references
  ./build/pyutils/perfdriver.py -vvv plot compare -i $references $resultdir/result.json -o plot-$domain.png || { echo 'Plotting failed'; rm -rf $tmpdir; exit 1; }
done

# clean possible temporary leftovers
rm -rf $tmpdir

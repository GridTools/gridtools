# use the machines python virtualenv with required modules (matplotlib) installed
source /project/c14/jenkins/python-venvs/${label%%-*}/bin/activate

# print full PATH for debugging purposes
echo "$PATH"

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
  resultdir=/project/c14/jenkins/gridtools-performance-history/$config/$grid/$precision/$backend/$domain
  mkdir -p $resultdir
  
  # name result file by date/time
  resultname=$(date +%F-%H-%M-%S).json
  
  # run performance tests
  ./build/pyutils/perfdriver.py -vvv run -d $domain $domain 80 --config $config -o $resultdir/$resultname || { echo 'Running failed'; rm -rf $tmpdir; exit 1; }
  
  # find previous results for history plot
  results=$(find $resultdir -regex '.*\.json')
  
  # plot history
  ./build/pyutils/perfdriver.py -vvv plot history -i $results -o history-$domain-full.png || { echo 'Plotting failed'; rm -rf $tmpdir; exit 1; }
  ./build/pyutils/perfdriver.py -vvv plot history -i $results -o history-$domain-last.png --limit=10 || { echo 'Plotting failed'; rm -rf $tmpdir; exit 1; }
done

# clean possible temporary leftovers
rm -rf $tmpdir

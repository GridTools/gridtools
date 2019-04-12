# use the machines python virtualenv with required modules installed
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

./pyutils/builddriver.py -vvv -b $build_type -p $real_type -g structured -d $target -c $compiler -o build -i install -t install || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }

./build/pyutils/testdriver.py -vvv -m -b || { echo 'Tests failed'; rm -rf $tmpdir; exit 2; }


# clean possible temporary leftovers
rm -rf $tmpdir

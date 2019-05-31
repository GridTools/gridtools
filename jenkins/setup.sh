#!/bin/bash

echo "Running on host $(hostname)"

# remove -cn from label (for daint)
label=${label%%-*}

envfile=./jenkins/envs/${label}_$env.sh

# use the machines python virtualenv with required modules installed
source /project/c14/jenkins/python-venvs/$label/bin/activate

if [[ $label != "kesch" ]]; then
    export SLURM_ACCOUNT=d75
    export SBATCH_ACCOUNT=d75
fi

# possibly delete old log files and create new log file
find /tmp -maxdepth 1 -mtime +5 -name 'gridtools-jenkins-*.log' -execdir rm -f {} +
logfile=$(mktemp -p /tmp gridtools-jenkins-XXXXX.log)
chmod +r $logfile

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

# register cleanup function
cleanup() {
    # clean possible temporary leftovers
    rm -rf $tmpdir
}

trap cleanup EXIT

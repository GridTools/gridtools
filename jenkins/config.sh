echo "Running on host $(hostname)"

# remove -cn from label (for daint)
label=${label%%-*}

envfile=./jenkins/envs/${label}_$env.sh

export GT_PERFORMANCE_HISTORY_PATH_PREFIX=/apps/${label}/SSL/gridtools/jenkins/gridtools-performance-history-gt2

# use the machines python virtualenv with required modules installed
if [[ $label = ault ]]; then
    venv_dir=/users/fthaler/public/jenkins/gridtools-venv
elif [[ $label = dom ]]; then
    venv_dir=/apps/daint/SSL/gridtools/jenkins/venv
else
    venv_dir=/apps/$label/SSL/gridtools/jenkins/venv
fi
source $venv_dir/bin/activate

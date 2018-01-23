#!/bin/bash

# installs git hook and adds forks of GT core developers

# run this script from the root directory
if [ ! -f $PWD/LICENSE ] ; then
    echo "you have to run the script ./tools/setup/setup.sh from the root of the sources"
    exit -1
fi

# install git hook
cp tools/git_hooks/pre-commit .git/hooks/pre-commit

# add upstream
git remote add upstream git@github.com:eth-cscs/gridtools.git
# add remotes of all core developers
repos=( anstaf cosunae havogt fthaler stefanmoosbrugger mbianco )
for i in "${repos[@]}"
do
    git remote add $i git@github.com:$i/gridtools.git
done


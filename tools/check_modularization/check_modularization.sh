#!/bin/bash

# no_dependency "<of_module>" "<on_module>"
# checks that <of_module> does not depend on <on_module>
# i.e. <of_module> does not include any file from <on_module>
function no_dependency() {
    last_result=`grep -r "#include .*$2/.*hpp" include/$1 | wc -l`
    if [ "$last_result" -gt 0 ]; then
        echo "ERROR Modularization violated: found dependency of $1 on $2"
        echo "`grep -r "#include .*$2/.*hpp" include/$1`"
    fi
    modularization_result=$(( modularization_result || last_result ))
}

function are_independent() {
    no_dependency "$1" "$2"
    no_dependency "$2" "$1"
}
modularization_result=0

# list of non-dependencies
no_dependency "common" "stencil-composition"
no_dependency "common" "boundary-conditions"
no_dependency "common" "communication"
no_dependency "common" "storage"
no_dependency "common" "tools"

are_independent "stencil-composition" "boundary-conditions"
are_independent "stencil-composition" "communication"
no_dependency "stencil-composition" "tools"

are_independent "boundary-conditions" "communication" #maybe they can have a dependency later?
are_independent "boundary-conditions" "tools"

no_dependency "communication" "storage"
#are_independent "communication" "storage" #TODO: violated by partitioner

no_dependency "storage" "stencil-composition"
no_dependency "storage" "boundary-conditions"
no_dependency "storage" "tools"

# we cannot use an exit code here because the git hook will terminate immediately

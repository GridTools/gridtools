#!/bin/bash

# no_dependency "<of_module>" "<on_module>"
# checks that <of_module> does not depend on <on_module>
# i.e. <of_module> does not include any file from <on_module>
function no_dependency() {
    last_result=`grep -r "#include .*$2/.*hpp" include/gridtools/$1 | wc -l`
    if [ "$last_result" -gt 0 ]; then
        echo "ERROR Modularization violated: found dependency of $1 on $2"
        echo "`grep -r "#include .*$2/.*hpp" include/gridtools/$1`"
    fi
    modularization_result=$(( modularization_result || last_result ))
}

function are_independent() {
    no_dependency "$1" "$2"
    no_dependency "$2" "$1"
}
modularization_result=0

# list of non-dependencies
no_dependency "common" "stencil_composition"
no_dependency "common" "boundary_conditions"
no_dependency "common" "communication"
no_dependency "common" "storage"
no_dependency "common" "tools"
no_dependency "common" "interface"
no_dependency "common" "c_bindings"
no_dependency "common" "distributed_boundaries"

are_independent "stencil_composition" "boundary_conditions"
are_independent "stencil_composition" "communication"
are_independent "stencil_composition" "interface"
no_dependency "stencil_composition" "tools"

are_independent "boundary_conditions" "communication" #maybe they can have a dependency later?
are_independent "boundary_conditions" "tools"
are_independent "boundary_conditions" "interface"

are_independent "communication" "interface"
are_independent "communication" "storage"

are_independent "distributed_boundaries" "c_bindings"
are_independent "distributed_boundaries" "stencil_composition"
are_independent "distributed_boundaries" "interface"
are_independent "distributed_boundaries" "tools"

no_dependency "storage" "stencil_composition"
no_dependency "storage" "boundary_conditions"
no_dependency "storage" "tools"
no_dependency "storage" "interface"

# we cannot use an exit code here because the git hook will terminate immediately

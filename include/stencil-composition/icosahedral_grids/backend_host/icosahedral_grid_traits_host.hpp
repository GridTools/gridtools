#pragma once
#include "execute_kernel_functor_host.hpp"

namespace gridtools {

    namespace icgrid {
        struct grid_traits_host {
            template < typename RunFunctorArguments >
            struct kernel_functor_executer {
                GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArguments >::value), "Error");
                typedef icgrid::execute_kernel_functor_host< RunFunctorArgs > type;
            }
        };
    }
}

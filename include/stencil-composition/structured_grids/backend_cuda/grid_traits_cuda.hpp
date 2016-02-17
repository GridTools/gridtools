#pragma once

#include "../../backend_cuda/execute_kernel_functor_cuda.hpp"

namespace gridtools {

    namespace strgrid {
        template<>
        struct grid_traits_arch<enumtype::Cuda> {
            template < typename RunFunctorArguments >
            struct kernel_functor_executer {
                GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArguments >::value), "Error");
                typedef execute_kernel_functor_cuda< RunFunctorArguments > type;
            };
        };
    }
}



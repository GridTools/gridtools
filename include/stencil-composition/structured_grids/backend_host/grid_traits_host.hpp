#pragma once

#include "execute_kernel_functor_host.hpp"
#include "execute_reduction_functor_host.hpp"

namespace gridtools {

    namespace strgrid {
        template<>
        struct grid_traits_arch<enumtype::Host> {
            template < typename RunFunctorArguments >
            struct kernel_functor_executer {
                GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArguments >::value), "Error");
                typedef execute_kernel_functor_host<RunFunctorArguments> type;
            };
        };
    }
}

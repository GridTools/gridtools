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
                typedef typename boost::mpl::eval_if<
                    typename RunFunctorArguments::is_reduction_t,
                    boost::mpl::identity<execute_reduction_functor_host<RunFunctorArguments> >,
                    boost::mpl::identity<execute_kernel_functor_host<RunFunctorArguments> >
                >::type type;
            };
        };
    }
}

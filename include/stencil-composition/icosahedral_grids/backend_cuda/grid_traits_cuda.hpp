/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once

#include "../grid_traits_backend_fwd.hpp"
#include "execute_kernel_functor_cuda_fwd.hpp"
#include "../../run_functor_arguments_fwd.hpp"

namespace gridtools {

    namespace icgrid {
        template <>
        struct grid_traits_arch< enumtype::Cuda > {
            template < typename RunFunctorArguments >
            struct kernel_functor_executor {
                GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArguments >::value), "Error");
                typedef execute_kernel_functor_cuda< RunFunctorArguments > type;
            };

            typedef layout_map< 3, 2, 1, 0 > layout_map_t;
        };
    }
}

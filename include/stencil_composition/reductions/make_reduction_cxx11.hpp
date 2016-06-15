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
#include <boost/make_shared.hpp>
#include "../esf.hpp"
#include "common/generic_metafunctions/variadic_to_vector.hpp"

namespace gridtools {

    template < typename RedFunctor, typename BinOp, typename ReductionType, typename... ExtraArgs >
    reduction_descriptor< ReductionType,
        BinOp,
        boost::mpl::vector1< esf_descriptor< RedFunctor, typename variadic_to_vector< ExtraArgs... >::type > > >
    make_reduction(const ReductionType initial_value, ExtraArgs...) {
#ifndef STRUCTURED_GRIDS
        GRIDTOOLS_STATIC_ASSERT((false), "Reductions are not yet supported for non structured grids");
#endif
#ifdef __CUDACC__
        GRIDTOOLS_STATIC_ASSERT((false), "Reductions are not yet supported for GPU backend");
#endif
        return reduction_descriptor< ReductionType,
            BinOp,
            boost::mpl::vector1< esf_descriptor< RedFunctor, typename variadic_to_vector< ExtraArgs... >::type > > >(
            initial_value);
    }

} // namespace gridtools

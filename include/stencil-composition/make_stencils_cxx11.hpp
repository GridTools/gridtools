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

#include "common/generic_metafunctions/variadic_to_vector.hpp"
#include "mss_metafunctions.hpp"
#include "mss.hpp"
#include "conditionals/if_.hpp"
#include "conditionals/case_.hpp"
#include "conditionals/switch_.hpp"

namespace gridtools {

    template < typename ExecutionEngine, typename... MssParameters >
    mss_descriptor< ExecutionEngine,
        typename extract_mss_esfs< typename variadic_to_vector< MssParameters... >::type >::type,
        typename extract_mss_caches< typename variadic_to_vector< MssParameters... >::type >::type >
    make_mss(ExecutionEngine && /**/, MssParameters...) {

        GRIDTOOLS_STATIC_ASSERT((is_execution_engine< ExecutionEngine >::value),
            "The first argument passed to make_mss must be the execution engine (e.g. execute<forward>(), "
            "execute<backward>(), execute<parallel>()");

        return mss_descriptor< ExecutionEngine,
            typename extract_mss_esfs< typename variadic_to_vector< MssParameters... >::type >::type,
            typename extract_mss_caches< typename variadic_to_vector< MssParameters... >::type >::type >();
    }

    template < typename... EsfDescr >
    independent_esf< boost::mpl::vector< EsfDescr... > > make_independent(EsfDescr &&...) {
        return independent_esf< boost::mpl::vector< EsfDescr... > >();
    }

} // namespace gridtools

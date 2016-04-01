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

#pragma once

#include "common/generic_metafunctions/variadic_to_vector.hpp"
#include "mss_metafunctions.hpp"
#include "mss.hpp"

namespace gridtools {

    template <typename ExecutionEngine,
        typename ... MssParameters >
    mss_descriptor<
        ExecutionEngine,
        typename extract_mss_esfs<typename variadic_to_vector<MssParameters ... >::type >::type,
        false,
        typename extract_mss_caches<typename variadic_to_vector<MssParameters ...>::type >::type
    >
    make_mss(ExecutionEngine&& /**/, MssParameters ...  ) {
        return mss_descriptor<
            ExecutionEngine,
            typename extract_mss_esfs<typename variadic_to_vector<MssParameters ... >::type >::type,
            false,
            typename extract_mss_caches<typename variadic_to_vector<MssParameters ... >::type >::type
        >();
    }


    template <typename ... EsfDescr >
    independent_esf< boost::mpl::vector<EsfDescr ...> >
    make_independent(EsfDescr&& ... ) {
        return independent_esf<boost::mpl::vector<EsfDescr... > >();
    }

} // namespace gridtools

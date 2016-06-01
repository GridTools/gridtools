#pragma once
#include "esf.hpp"

namespace gridtools {

    template < uint_t T, uint_t SwitchId >
    struct conditional;

    template < typename EsfSequence >
    struct independent_esf {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< EsfSequence, is_esf_descriptor >::value),
            "Error: independent_esf requires a sequence of esf's");
        typedef EsfSequence esf_list;
    };

    template < typename T >
    struct is_independent : boost::false_type {};

    template < typename T >
    struct is_independent< independent_esf< T > > : boost::true_type {};

} // namespace gridtools

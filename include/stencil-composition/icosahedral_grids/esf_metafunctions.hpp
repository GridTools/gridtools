#pragma once
#include <boost/mpl/equal.hpp>

namespace gridtools {
    namespace icgrid {
        template < typename EsfSequence >
        struct extract_esf_location_type {
            GRIDTOOLS_STATIC_ASSERT(
                (is_sequence_of< EsfSequence, is_esf_descriptor >::value), "Error, wrong esf types");
            typedef typename boost::mpl::fold< EsfSequence,
                boost::mpl::set0<>,
                boost::mpl::insert< boost::mpl::_1, esf_get_location_type< boost::mpl::_2 > > >::type
                location_type_set_t;
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< location_type_set_t >::value == 1),
                "Error: all ESFs should have the same location type");

            typedef typename boost::mpl::front< location_type_set_t >::type type;
        };
    }

    template < typename Esf1, typename Esf2 >
    struct esf_equal {
        GRIDTOOLS_STATIC_ASSERT(
            (is_esf_descriptor< Esf1 >::value && is_esf_descriptor< Esf2 >::value), "Error: Internal Error");
        typedef static_bool< boost::is_same< typename Esf1::esf_function, typename Esf2::esf_function >::value &&
                             boost::mpl::equal< typename Esf1::args_t, typename Esf2::args_t >::value &&
                             boost::is_same< typename Esf1::location_type, typename Esf2::location_type >::value &&
                             boost::is_same< typename Esf1::grid_t, typename Esf2::grid_t >::value > type;
    };

} // namespace gridtools

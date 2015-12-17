#pragma once

namespace gridtools {
    namespace icgrid {
        template<typename EsfSequence>
        struct extract_esf_location_type
        {
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of<EsfSequence, is_esf_descriptor>::value),
                                    "Error, wrong esf types");
            typedef typename boost::mpl::fold<
                EsfSequence,
                boost::mpl::set0<>,
                boost::mpl::insert<boost::mpl::_1, esf_get_location_type<boost::mpl::_2> >
            >::type location_type_set_t;
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<location_type_set_t>::value == 1),
                                    "Error: all ESFs should have the same location type");

            typedef typename boost::mpl::front<location_type_set_t>::type type;
        };
    }
} //namespace gridtools

#pragma once
#include "../mss.hpp"
#include "../mss_metafunctions.hpp"
#include "../amss_descriptor.hpp"

namespace gridtools {

    namespace icgrid {

        template < typename MssDescriptor >
        struct mss_compute_extent_sizes {
            GRIDTOOLS_STATIC_ASSERT((is_mss_descriptor< MssDescriptor >::value), "Internal Error: invalid type");

            /**
             * for rectangular grids pushing null extents
             */
            typedef typename boost::mpl::fold< typename mss_descriptor_esf_sequence< MssDescriptor >::type,
                boost::mpl::vector0<>,
                boost::mpl::push_back< boost::mpl::_1, extent< 0 > > >::type type;

            GRIDTOOLS_STATIC_ASSERT(
                (boost::mpl::size< typename mss_descriptor_linear_esf_sequence< MssDescriptor >::type >::value ==
                    boost::mpl::size< type >::value),
                "Internal Error: wrong size");
        };
    }
}

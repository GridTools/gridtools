#pragma once

#include <boost/mpl/max.hpp>

namespace gridtools {

    template < int_t R = 0 >
    struct extent {
        static const int_t value = R;
    };

    template < typename T >
    struct is_extent : boost::mpl::false_ {};

    template < int R >
    struct is_extent< extent< R > > : boost::mpl::true_ {};

    /**
     * Metafunction taking two extents and yielding a extent containing them
     */
    template < typename Extent1, typename Extent2 >
    struct enclosing_extent {
        BOOST_MPL_ASSERT((is_extent< Extent1 >));
        BOOST_MPL_ASSERT((is_extent< Extent2 >));

        typedef extent< boost::mpl::max< static_uint< Extent1::value >, static_uint< Extent2::value > >::type::value >
            type;
    };

    /**
     * Metafunction to add two extents
     */
    template < typename Extent1, typename Extent2 >
    struct sum_extent {
        BOOST_MPL_ASSERT((is_extent< Extent1 >));
        BOOST_MPL_ASSERT((is_extent< Extent2 >));

        typedef extent< Extent1::value + Extent2::value > type;
    };

} // namespace gridtools

#pragma once

namespace gridtools {

    template < int R >
    struct extent {
        static const int value = R;
    };

    template < typename T >
    struct is_extent : boost::mpl::false_ {};

    template < int R >
    struct is_extent< extent< R > > : boost::mpl::true_ {};

    template < typename T >
    struct undef_t;
    /**
     * Metafunction taking two extents and yielding a extent containing them
     */
    template < typename Extent1, typename Extent2 >
    struct enclosing_extent {
        BOOST_MPL_ASSERT((is_extent< Extent1 >));
        BOOST_MPL_ASSERT((is_extent< Extent2 >));

        typedef extent< boost::mpl::max< typename Extent1::value, typename Extent2::value >::type::value > type;
    };

} // namespace gridtools

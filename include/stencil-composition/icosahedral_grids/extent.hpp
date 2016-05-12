#pragma once

namespace gridtools {

    template < int_t R=0 >
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

    template < typename Range1 >
    struct enclosing_extent<Range1, empty_extent> {
        typedef typename enclosing_extent<Range1, extent<> >::type type;
    };

    template < typename Range2 >
    struct enclosing_extent<empty_extent, Range2> {
        typedef typename enclosing_extent<extent<>, Range2>::type type;
    };

    template <  >
    struct enclosing_extent<empty_extent, empty_extent> {
        typedef typename enclosing_extent<extent<>, extent<> >::type type;
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


    template < typename Range1 >
    struct sum_extent<Range1, empty_extent> {
        typedef typename sum_extent<Range1, extent<> >::type type;
    };

    template < typename Range2 >
    struct sum_extent<empty_extent, Range2> {
        typedef typename sum_extent<extent<>, Range2>::type type;
    };

    template <  >
    struct sum_extent<empty_extent, empty_extent> {
        typedef typename sum_extent<extent<>, extent<> >::type type;
    };

} // namespace gridtools

#pragma once

namespace gridtools {
    template < typename Extent1 >
    struct sum_extent< Extent1, empty_extent > {
        GRIDTOOLS_STATIC_ASSERT((is_extent< Extent1 >::value), "Type should be an Extent");
        typedef typename sum_extent< Extent1, extent<> >::type type;
    };

    template < typename Extent2 >
    struct sum_extent< empty_extent, Extent2 > {
        GRIDTOOLS_STATIC_ASSERT((is_extent< Extent2 >::value), "Type should be an Extent");
        typedef typename sum_extent< extent<>, Extent2 >::type type;
    };

    template <>
    struct sum_extent< empty_extent, empty_extent > {
        typedef typename sum_extent< extent<>, extent<> >::type type;
    };

    /**
     * Metafunction to check if a type is a extent - Specialization yielding true
     */
    template <>
    struct is_extent< empty_extent > : boost::true_type {};

    template < typename Extent1 >
    struct enclosing_extent< Extent1, empty_extent > {
        GRIDTOOLS_STATIC_ASSERT((is_extent< Extent1 >::value), "Type should be an Extent");
        typedef typename enclosing_extent< Extent1, extent<> >::type type;
    };

    template < typename Extent2 >
    struct enclosing_extent< empty_extent, Extent2 > {
        GRIDTOOLS_STATIC_ASSERT((is_extent< Extent2 >::value), "Type should be an Extent");
        typedef typename enclosing_extent< extent<>, Extent2 >::type type;
    };

    template <>
    struct enclosing_extent< empty_extent, empty_extent > {
        typedef typename enclosing_extent< extent<>, extent<> >::type type;
    };

} //namespace gridtools

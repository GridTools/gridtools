#pragma once

namespace gridtools {
    template < uint_t X, uint_t Y
#ifdef CXX11_ENABLED
, uint_t ... Rest
#else
, uint_t Z=0
#endif
>
    struct block_size {
        typedef boost::mpl::integral_c< int, X > i_size_t;
        typedef boost::mpl::integral_c< int, Y > j_size_t;
        typedef boost::mpl::integral_c< int, Z > k_size_t;
    };

    template < typename T >
    struct is_block_size : boost::mpl::false_ {};

    template < uint_t X, uint_t Y, uint_t Z
#ifdef CXX11_ENABLED
               , uint_t ... Rest
#endif
               >
    struct is_block_size< block_size< X, Y, Z
#ifdef CXX11_ENABLED
                                      , Rest ...
#endif
                                      > > : boost::mpl::true_ {};
} // namespace gridtools

#pragma once

namespace gridtools {
    template<uint_t X, uint_t Y>
    struct block_size
    {
        typedef boost::mpl::integral_c<int, X> i_size_t;
        typedef boost::mpl::integral_c<int, Y> j_size_t;
    };

    template<typename T>
    struct is_block_size : boost::mpl::false_{};

    template<uint_t X, uint_t Y>
    struct is_block_size<block_size<X, Y> > : boost::mpl::true_{};
} // namespace gridtools

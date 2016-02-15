#pragma once

namespace gridtools{

    template <int R>
    struct extent {
        static const int value = R;
    };

    template<typename T> struct is_extent: boost::mpl::false_{};

    template<int R> struct is_extent<extent<R> > : boost::mpl::true_{};
} // namespace gridtools

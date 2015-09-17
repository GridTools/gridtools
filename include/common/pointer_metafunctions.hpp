#pragma once
#include<boost/mpl/bool.hpp>

namespace gridtools{
    template<typename T>
    struct is_pointer : boost::mpl::false_{};

    template<typename T>
    struct is_pointer<pointer<T> > : boost::mpl::true_{};
}

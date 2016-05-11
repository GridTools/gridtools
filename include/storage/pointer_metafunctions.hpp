#pragma once

namespace gridtools {

    template <typename T, bool B>
    struct hybrid_pointer;

    template <typename T>
    struct is_hybrid_pointer : boost::mpl::false_ {};

    template <typename T>
    struct is_hybrid_pointer<hybrid_pointer<T, false> > : boost::mpl::true_ {};
    template <typename T>
    struct is_hybrid_pointer<hybrid_pointer<T, true> > : boost::mpl::true_ {};


    template <typename T, bool B>
    struct wrap_pointer;

    template <typename T>
    struct is_wrap_pointer : boost::mpl::false_ {};

    template <typename T>
    struct is_wrap_pointer<wrap_pointer<T, false> > : boost::mpl::true_ {};
    template <typename T>
    struct is_wrap_pointer<wrap_pointer<T, true> > : boost::mpl::true_ {};

}

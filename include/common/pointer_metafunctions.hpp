#pragma once
#include<boost/mpl/bool.hpp>

namespace gridtools{

    /** \addtogroup specializations Specializations
        Partial specializations
        @{
    */
    template<typename T>
    struct is_pointer : boost::mpl::false_{};

    template<typename T>
    struct is_pointer<pointer<T> > : boost::mpl::true_{};

    template<typename T>
    struct is_ptr_to_tmp : boost::mpl::false_{};

    template<typename T>
    struct is_ptr_to_tmp<pointer<const T> > : boost::mpl::bool_<T::is_temporary> {};
    /**@}*/

}

#pragma once
#include <boost/type_traits.hpp>
#include <boost/mpl/bool.hpp>

namespace gridtools {
    template <typename T>
    struct remove_ref_cv {
        typedef typename boost::remove_reference< 
            typename boost::remove_cv<T>::type 
        >::type type;
    };

    /** \addtogroup specializations Specializations
        Partial specializations
        @{
    */
    template < typename T >
    struct is_pointer_impl : boost::mpl::false_ {};

    template < typename T >
    struct is_pointer_impl< pointer< T > > : boost::mpl::true_ {};

    template < typename T >
    struct is_pointer : is_pointer_impl< typename remove_ref_cv<T>::type > {};


    template < typename T >
    struct is_ptr_to_tmp_impl : boost::mpl::false_ {};

    template < typename T >
    struct is_ptr_to_tmp_impl< pointer< const T > > : boost::mpl::bool_< T::is_temporary > {};

    template < typename T >
    struct is_ptr_to_tmp : is_ptr_to_tmp_impl< typename remove_ref_cv<T>::type > {};
    /**@}*/

    template <typename T, bool B>
    struct hybrid_pointer;

    template <typename T>
    struct is_hybrid_pointer_impl : boost::mpl::false_ {};

    template <typename T>
    struct is_hybrid_pointer_impl<hybrid_pointer<T, false> > : boost::mpl::true_ {};

    template <typename T>
    struct is_hybrid_pointer_impl<hybrid_pointer<T, true> > : boost::mpl::true_ {};

    template <typename T>
    struct is_hybrid_pointer : is_hybrid_pointer_impl< typename remove_ref_cv<T>::type > {};


    template <typename T, bool B>
    struct wrap_pointer;

    template <typename T>
    struct is_wrap_pointer_impl : boost::mpl::false_ {};

    template <typename T>
    struct is_wrap_pointer_impl<wrap_pointer<T, false> > : boost::mpl::true_ {};

    template <typename T>
    struct is_wrap_pointer_impl<wrap_pointer<T, true> > : boost::mpl::true_ {};

    template <typename T>
    struct is_wrap_pointer : is_wrap_pointer_impl< typename remove_ref_cv<T>::type > {};

}

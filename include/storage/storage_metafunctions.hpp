#pragma once

/**
   @file
   @brief File containing a set of metafunctions that apply to the storage classes.
*/

#include "storage.hpp"

namespace gridtools {

    /**
     * @brief The storage_holds_data_field struct
     * determines if the storage class is holding a data field type of storage
     */
    template < typename T >
    struct storage_holds_data_field : boost::mpl::bool_< (T::field_dimensions > 1) > {};

    /**@brief metafunction to extract the metadata type from a storage pointer

    */
    template < typename Storage >
    struct storage2metadata;

    template < typename Storage >
    struct storage2metadata< pointer< Storage > > {
        typedef typename Storage::storage_info_type type;
    };

    /**
       \addtogroup specializations Specializations
       @{
    */
    template < typename U >
    struct is_temporary_storage< no_storage_type_yet< U > > : public boost::true_type {};

    template < typename T, typename U, ushort_t Dim >
    struct is_actual_storage< pointer< base_storage< T, U, Dim > > > : public boost::mpl::bool_< !U::is_temporary > {};

    template < typename U >
    struct is_actual_storage< pointer< no_storage_type_yet< U > > > : public boost::false_type {};

    template < typename U >
    struct is_temporary_storage< pointer< no_storage_type_yet< U > > > : public boost::true_type {};

    // Decorator is e.g. of type storage
    template < typename BaseType, template < typename T > class Decorator >
    struct is_actual_storage< pointer< Decorator< BaseType > > >
        : public is_actual_storage< pointer< typename BaseType::basic_type > > {};

#ifdef CXX11_ENABLED
    // Decorator is e.g. a data_field
    template < typename First, typename... BaseType, template < typename... T > class Decorator >
    struct is_actual_storage< pointer< Decorator< First, BaseType... > > >
        : public is_actual_storage< pointer< typename First::basic_type > > {};

#else

    // Decorator is the integrator
    template < typename First,
        typename B2,
        typename B3,
        template < typename T1, typename T2, typename T3 > class Decorator >
    struct is_actual_storage< pointer< Decorator< First, B2, B3 > > >
        : public is_actual_storage< pointer< typename First::basic_type > > {};

#endif

    // Decorator is the integrator
    template < typename BaseType, template < typename T, ushort_t O > class Decorator, ushort_t Order >
    struct is_actual_storage< Decorator< BaseType, Order > * >
        : public is_actual_storage< typename BaseType::basic_type * > {};

    template < typename T >
    struct is_any_storage : is_storage< T > {};

    template < typename T >
    struct is_any_storage< no_storage_type_yet< T > > : boost::mpl::true_ {};

    template < typename T >
    struct is_any_storage< pointer< T > > : is_storage< T > {};

    template < typename T >
    struct is_any_storage< pointer< no_storage_type_yet< T > > > : boost::mpl::true_ {};

    template < typename T >
    struct is_not_tmp_storage : boost::mpl::or_< is_actual_storage< T >, boost::mpl::not_< is_any_storage< T > > > {};

    template < typename T >
    struct storage_pointer_type {
        typedef typename T::pointer_type type;
    };
}

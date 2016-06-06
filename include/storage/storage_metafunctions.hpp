#pragma once

/**
   @file
   @brief File containing a set of metafunctions that apply to the storage classes.
*/
#include <boost/type_traits.hpp>

#include "storage.hpp"
#include "base_storage.hpp"
#include "metadata_set.hpp"

namespace gridtools {

    // forward decl to global parameter
    template <typename D>
    struct global_parameter;

    /**@brief metafunction to check if a given type is a global parameter
    */
    template <typename T>
    struct is_global_parameter : boost::mpl::false_ { };

    template <typename T>
    struct is_global_parameter< global_parameter<T> > : boost::mpl::true_ { };

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

    template < typename BaseType >
    struct is_actual_storage< pointer< global_parameter< BaseType > > >
        : public is_actual_storage< pointer< BaseType > > {};

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

    /* @brief metafunction that takes a pointer<storage<base>> type and returns a pointer<base> type */
    template < typename T >
    struct get_user_storage_base_t {
        GRIDTOOLS_STATIC_ASSERT((is_pointer< T >::value), "the passed type is not a pointer type");
        GRIDTOOLS_STATIC_ASSERT((is_any_storage< typename T::value_type >::value),
            "the passed pointer type does not contain a storage type");
        typedef pointer< typename T::value_type::basic_type > type;
    };

    /** @brief metafunction that takes a pointer<storage<T>> type and returns a pointer<storage<T>::storage_ptr_t> type
     */
    template < typename T >
    struct get_user_storage_ptrs_t {
        typedef typename boost::remove_reference< T >::type ty;
        GRIDTOOLS_STATIC_ASSERT((is_pointer< ty >::value), "the passed type is not a pointer type");
        typedef typename ty::value_type storage_ty;
        GRIDTOOLS_STATIC_ASSERT(
            (is_any_storage< storage_ty >::value), "the passed pointer type does not contain a storage type");
        typedef typename storage_ty::storage_ptr_t storage_ptr_ty;
        GRIDTOOLS_STATIC_ASSERT(
            (is_hybrid_pointer< storage_ptr_ty >::value || is_wrap_pointer< storage_ptr_ty >::value),
            "the contained storage pointer type is neither a wrap nor a hybrid pointer type");
        typedef pointer< storage_ptr_ty > type;
    };

    /** @brief metafunction class that is used to transform a fusion vector of pointer<storage<T>> into a
     *  pointer<storage<T>::storage_ptr_t> vector
     */
    struct get_user_storage_ptrs {
        template < typename T >
        struct result;

        template < typename F, typename T >
        struct result< F(T) > {
            typedef typename get_user_storage_ptrs_t< T >::type type;
        };

        template < typename T >
        struct result< pointer< storage< T > > > {
            typedef typename get_user_storage_ptrs_t< pointer< storage< T > > >::type type;
        };

        template < typename T >
        typename get_user_storage_ptrs_t< T >::type operator()(T &st) const {
            typedef typename get_user_storage_ptrs_t< T >::type ty;
            return st->get_storage_pointer();
        }
    };

    /** @brief metafunction class that is used to extract metadata pointers from a fusion vector of pointer<storage<T>>
     */
    template <typename U>
    struct get_storage_metadata_ptrs {
        U& metadata_set;
        GRIDTOOLS_STATIC_ASSERT(is_metadata_set<U>::value, "passed type is not a metadata_set");
        get_storage_metadata_ptrs(U& ms) : metadata_set(ms) {}

        /** @brief overload for the case that the "storage" is a global_parameter. Skip the element in this case.
         */
        template <typename T>
        void operator()(T &st, typename boost::disable_if< is_global_parameter<typename T::value_type> >::type *a = 0) const {
            GRIDTOOLS_STATIC_ASSERT(is_any_storage<typename T::value_type>::value, "passed object is neither a pointer<storage<T>> nor a pointer<global_parameter<T>>");
            metadata_set.insert(st->get_meta_data_pointer());
        }

        /** @brief overload for the case that the "storage" is a global_parameter. Skip the element in this case.
         */
        template <typename T>
        void operator()(T &st, typename boost::enable_if< is_global_parameter<typename T::value_type> >::type *a = 0) const {}
    };
}

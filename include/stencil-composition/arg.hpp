/**
   @file
   @brief File containing the definition of the placeholders used to address the storage from whithin the functors.
   A placeholder is an implementation of the proxy design pattern for the storage class, i.e. it is a light object used in place of the storage when defining the high level computations,
   and it will be bound later on with a specific instantiation of a storage class.
*/

#pragma once
#include "storage/storage_metafunctions.hpp"
#include "stencil-composition/arg_metafunctions_fwd.hpp"

namespace gridtools {

    //fwd decl
    template <typename T> struct is_arg;

/** @brief binding between the placeholder (\tparam ArgType) and the storage (\tparam Storage)*/
template<typename ArgType, typename Storage>
struct arg_storage_pair {

    //TODO is_storage is taken!
    GRIDTOOLS_STATIC_ASSERT(is_arg<ArgType>::value, "wrong type");
    typedef ArgType arg_type;
    typedef Storage storage_type;

    Storage *ptr;

    arg_storage_pair(Storage* p)
        : ptr(p)
        {}

    Storage* operator*() {
        return ptr;
    }
};


    template<typename T>
    struct is_arg_storage_pair : boost::mpl::false_{};

    template<typename ArgType, typename Storage>
    struct is_arg_storage_pair<arg_storage_pair<ArgType, Storage> > : boost::mpl::true_{};

/**
 * Type to create placeholders for data fields.
 *
 * There is a specialization for the case in which T is a temporary
 *
 * @tparam I Integer index (unique) of the data field to identify it
 * @tparam T The type of the storage used to store data
 */
template <uint_t I, typename Storage>
struct arg {
    typedef Storage storage_type;
    typedef typename Storage::iterator_type iterator_type;
    typedef typename Storage::value_type value_type;
    typedef static_uint<I> index_type;
    typedef static_uint<I> index;

//location type is only used by other grids, supported only for cxx11
#ifdef CXX11_ENABLED
    using location_type = typename Storage::meta_data_t::index_type;
#endif
     
    template<typename Storage2>
    arg_storage_pair<arg<I,Storage>, Storage2>
    operator=(Storage2& ref) {
        GRIDTOOLS_STATIC_ASSERT( (boost::is_same<Storage2, Storage>::value), "there is a mismatch between the storage types used by the arg placeholders and the storages really instantiated. Check that the placeholders you used when constructing the domain_type are in the correctly assigned and that their type match the instantiated storages ones" );

        return arg_storage_pair<arg<I,Storage>, Storage2>(&ref);
    }

    static void info() {
#ifdef VERBOSE
        std::cout << "Arg on real storage with index " << I;
#endif
    }
};

template<typename T>
struct is_arg : boost::mpl::false_{};

template<uint_t I, typename Storage>
struct is_arg<arg<I, Storage> > : boost::mpl::true_{};

template<typename T>
struct arg_index;

/** true in case of non temporary storage arg*/
template<uint_t I, typename Storage>
struct arg_index<arg<I, Storage> > : boost::mpl::integral_c<int, I> {};

template<typename T>
struct is_storage_arg : boost::mpl::false_{};

template<uint_t I, typename Storage>
struct is_storage_arg<arg<I, Storage> > : is_storage<Storage>{};

/** true in case of non temporary storage arg*/
template<uint_t I, typename Storage>
struct arg_index<arg<I, Storage> > : boost::mpl::integral_c<int, I> {};

/**
 * @struct arg_hods_data_field
 * metafunction that determines if an arg type is holding the storage type of a data field
 */
template<typename Arg>
struct arg_holds_data_field;

template <uint_t I, typename Storage>
struct arg_holds_data_field<arg<I, Storage> >
{
    typedef typename storage_holds_data_field<Storage>::type type;
};

/**
 * @struct arg_hods_data_field_h
 * high order metafunction of arg_holds_data_field
 */
template <typename Arg>
struct arg_holds_data_field_h
{
    typedef typename arg_holds_data_field<typename Arg::type >::type type;
};

/** @brief metafunction to access the storage type given the arg*/
template<typename T>
struct arg2storage {
    GRIDTOOLS_STATIC_ASSERT(is_arg<T>::value, "wrong type for Arg");
    typedef typename T::storage_type type;
};

/** @brief metafunction to access the metadata type given the arg*/
template<typename T>
struct arg2metadata {
    GRIDTOOLS_STATIC_ASSERT(is_arg<T>::value, "wrong type for Arg");
    typedef typename arg2storage<T>::type::meta_data_t type;
};


} // namespace gridtools

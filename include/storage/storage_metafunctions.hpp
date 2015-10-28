#pragma once

/**
   @file
   @brief File containing a set of metafunctions that apply to the storage classes.
*/

#include "storage.hpp"
#include "base_storage.hpp"

namespace gridtools{

/**
 * @brief The storage_holds_data_field struct
 * determines if the storage class is holding a data field type of storage
 */
template<typename T>
struct storage_holds_data_field : boost::mpl::false_{};


#ifdef CXX11_ENABLED
template <typename First,  typename  ...  StorageExtended>
struct storage_holds_data_field<storage<data_field<First, StorageExtended ... > > > : boost::mpl::true_ {};
#endif


    /**@brief metafunction to extract the metadata from a storage

    */
    template<typename Storage>
    struct storage2metadata{
        typedef typename Storage::meta_data_t
        type;
    };


    /**
       \addtogroup specializations Specializations
       @{
    */
    template <typename U>
    struct is_temporary_storage<no_storage_type_yet<U>  > : public boost::true_type
    {};

    template <typename T, typename U, ushort_t Dim>
    struct is_storage<base_storage<T,U,Dim>  *  > : public boost::mpl::bool_< !U::is_temporary >
    {};

    template <typename U>
    struct is_storage<no_storage_type_yet<U>  *  > : public boost::false_type
    {};

    template <typename U>
    struct is_temporary_storage<no_storage_type_yet<U>* > : public boost::true_type
    {};

    template <typename U>
    struct is_temporary_storage<no_storage_type_yet<U>& > : public boost::true_type
    {};

    //Decorator is the storage
    template <typename BaseType , template <typename T> class Decorator >
    struct is_storage<Decorator<BaseType>  *  > : public is_storage<typename BaseType::basic_type*>
    {};

    //Decorator is the storage
    template <typename BaseType , template <typename T> class Decorator >
    struct is_storage<Decorator<BaseType> > : public is_storage<typename BaseType::basic_type*>
    {};

#ifdef CXX11_ENABLED
    //Decorator is the integrator
    template <typename First, typename ... BaseType , template <typename ... T> class Decorator >
    struct is_storage<Decorator<First, BaseType...>  *  > : public is_storage<typename First::basic_type*>
    {};
#else

    //Decorator is the integrator
    template <typename First, typename B2, typename  B3 , template <typename T1, typename T2, typename T3> class Decorator >
    struct is_storage<Decorator<First, B2, B3>  *  > : public is_storage<typename First::basic_type*>
    {};

#endif

    //Decorator is the integrator
    template <typename BaseType , template <typename T, ushort_t O> class Decorator, ushort_t Order >
    struct is_storage<Decorator<BaseType, Order>  *  > : public is_storage<typename BaseType::basic_type*>
    {};

    template <typename T>
    struct is_any_storage : boost::mpl::false_{};

    template<typename T>
    struct is_any_storage<storage<T> > : boost::mpl::true_{};

    template<typename T>
    struct is_any_storage<no_storage_type_yet<T> > : boost::mpl::true_{};

    template<typename T>
    struct is_any_storage<storage<T>* > : boost::mpl::true_{};

    template<typename T>
    struct is_any_storage<storage<T>*& > : boost::mpl::true_{};

    template<typename T>
    struct is_any_storage<no_storage_type_yet<T>* > : boost::mpl::true_{};

    template<typename T>
    struct is_any_storage<no_storage_type_yet<T>*& > : boost::mpl::true_{};

    template<typename T>
    struct is_not_tmp_storage : boost::mpl::or_<is_storage<T>, boost::mpl::not_<is_any_storage<T > > >{
    };


}

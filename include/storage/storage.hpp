#pragma once
#include "data_field.hpp"
#include "meta_storage.hpp"
#include "common/gpu_clone.hpp"
#include "common/generic_metafunctions/reverse_pack.hpp"

/**
@file
@brief Storage class
This extra layer is added on top of the base_storage class because it extends the clonabl_to_gpu interface. Due to the multiple inheritance pattern this class should not be further inherited.
*/

namespace gridtools{

    template < typename BaseStorage >
      struct storage : public BaseStorage, clonable_to_gpu<storage<BaseStorage> >
    {
        typedef BaseStorage super;
        typedef typename BaseStorage::basic_type basic_type;
        typedef storage<BaseStorage> original_storage;
        typedef clonable_to_gpu<storage<BaseStorage> > gpu_clone;
        typedef typename BaseStorage::iterator_type iterator_type;
        typedef typename BaseStorage::value_type value_type;
        static const ushort_t n_args = basic_type::n_width;
        static const ushort_t space_dimensions = basic_type::space_dimensions;

      __device__
      storage(storage const& other)
          :  super(other)
      {}

#if defined(CXX11_ENABLED)
        //forwarding constructor
        template <class ... ExtraArgs>
        explicit storage(  typename basic_type::meta_data_t const& meta_data_, ExtraArgs const& ... args ):super(meta_data_, args ...)
            {
            }
#else
        template <class T>
        explicit storage(  typename basic_type::meta_data_t const& meta_data_, T const& arg1 ):super(meta_data_, arg1)
            {
            }

        template <class T, class U>
        explicit storage(  typename basic_type::meta_data_t const& meta_data_, T const& arg1, U const& arg2 ):super(meta_data_, arg1, arg2)
            {
            }

#endif

//    private :
        explicit storage(typename basic_type::meta_data_t const& meta_data_):super(meta_data_){}
    };

    /**@brief Convenient syntactic sugar for specifying an extended-dimension with extended-width storages, where each dimension has arbitrary size 'Number'.

       Annoyngly enough does not work with CUDA 6.5
    */
#if defined(CXX11_ENABLED)

    /** @brief syntactic sugar for defining a data field

        Given a storage type and the dimension number it generates the correct data field type
        @tparam Storage the basic storage used
        @tparam Number the number of snapshots in each dimension
     */
    template< class Storage, uint_t ... Number >
    struct field_reversed;

    /**
     @brief specialization for the GPU storage
     the defined type is storage (which is clonable_to_gpu)
    */
    template< class BaseStorage, uint_t ... Number >
    struct field_reversed<storage<BaseStorage>, Number ... >{
        typedef storage< data_field< storage_list<base_storage<typename BaseStorage::pointer_type, typename  BaseStorage::meta_data_t, accumulate(add_functor(), ((uint_t)Number) ... )>, Number-1> ... > > type;
    };

    /**
       @brief specialization for the CPU storage (base_storage)
       the difference being that the type is directly the base_storage (which is not clonable_to_gpu)
    */
    template< class PointerType, class MetaData, ushort_t FD, uint_t ... Number >
    struct field_reversed<base_storage<PointerType, MetaData, FD>, Number ... >{
        typedef data_field< storage_list<base_storage<PointerType, MetaData, accumulate(add_functor(), ((uint_t)Number) ... )>, Number-1> ... > type;
    };

    /**@brief specialization for no_storage_type_yet (Block strategy, GPU storage)*/
    template<  typename PointerType
               ,typename MetaData
               ,ushort_t FieldDimension
               ,uint_t ... Number >
    struct field_reversed<no_storage_type_yet<storage<base_storage<PointerType, MetaData, FieldDimension> > >, Number... >{
        typedef no_storage_type_yet<storage<data_field< storage_list<base_storage<PointerType, MetaData, accumulate(add_functor(), ((uint_t)Number) ... ) >, Number-1> ... > > > type;
    };

    /**@brief specialization for no_storage_type_yet (Block strategy, CPU storage)*/
    template<  typename PointerType
               ,typename MetaData
               ,ushort_t FieldDimension
               ,uint_t ... Number >
    struct field_reversed<no_storage_type_yet<base_storage<PointerType, MetaData, FieldDimension> >, Number... >{
        typedef no_storage_type_yet<data_field< storage_list<base_storage<PointerType, MetaData, accumulate(add_functor(), ((uint_t)Number) ... ) >, Number-1> ... > > type;
    };

    /**@brief interface for definig a data field

       @tparam Storage the basic storage type shared by all the snapshots
       @tparam First  all the subsequent parameters define the dimensionality of the snapshot arrays
        in all the data field dimensions
     */
    template< class Storage, uint_t First, uint_t ... Number >
    struct field{
        typedef typename reverse_pack<Number ...>::template apply<field_reversed, Storage, First >::type::type type;
    };
#endif

    template <typename T>
    std::ostream& operator<<(std::ostream &s, storage<T> const & x ) {
        s << "storage< "
          << static_cast<T const&>(x) << " > ";
        return s;
    }

}//namespace gridtools

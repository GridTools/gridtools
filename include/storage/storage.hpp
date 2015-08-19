#pragma once
#include "data_field.hpp"
#include "common/gpu_clone.hpp"
//#include "host_tmp_storage.hpp"
#include "accumulate.hpp"
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
#if defined(CXX11_ENABLED) //&& !defined(__CUDACC__)

    template< class Storage, uint_t ... Number >
    struct field_reversed{
        typedef storage< data_field< storage_list<base_storage<typename Storage::pointer_type, typename  Storage::meta_data_t, accumulate(add_functor(), ((uint_t)Number) ... )>, Number-1> ... > > type;
    };


    // // specialization for temporary storages/naive strategy

    // template<  typename PointerType
    //            ,typename MetaData
    //            ,short_t FieldDimension
    //            ,uint_t ... Number >
    // struct field_reversed<base_storage<PointerType, MetaData, FieldDimension>, Number... >{
    //     typedef storage< data_field< storage_list<base_storage<PointerType, MetaData, accumulate(add_functor(), ((uint_t)Number) ... )>, Number-1> ... > > type;
    // };

    // specialization for temporary storages/block strategy

    template<  typename PointerType
               ,typename MetaData
               ,short_t FieldDimension
               ,uint_t ... Number >
    struct field_reversed<no_storage_type_yet<storage<base_storage<PointerType, MetaData, FieldDimension> > >, Number... >{
        typedef no_storage_type_yet<storage<data_field< storage_list<base_storage<PointerType, MetaData, accumulate(add_functor(), ((uint_t)Number) ... ) >, Number-1> ... > > > type;
    };

    template< class Storage, uint_t First, uint_t ... Number >
    struct field{
        typedef typename reverse_pack<Number ...>::template apply<field_reversed, Storage, First >::type::type type;
    };

#else//CXX11_ENABLED


    template< class Storage, uint_t Number1, uint_t Number2, uint_t Number3 >
    struct field{
        typedef storage< data_field< storage_list<base_storage<typename Storage::pointer_type, typename  Storage::meta_data_t, Number1+Number2+Number3>, Number1-1>, storage_list<base_storage<typename Storage::pointer_type, typename  Storage::meta_data_t, Number1+Number2+Number3>, Number2-1>, storage_list<base_storage<typename Storage::pointer_type, typename  Storage::meta_data_t, Number1+Number2+Number3>, Number3-1> > > type;
    };


    template< class Storage, uint_t Number1>
    struct field1{
        typedef storage< data_field1< storage_list<base_storage<typename Storage::pointer_type, typename  Storage::meta_data_t, Number1>, Number1-1> > > type;
    };



    // template<  typename PointerType
    //            ,typename MetaData
    //            ,short_t FieldDimension
    //            , uint_t Number1, uint_t Number2, uint_t Number3 >
    // struct field<base_storage<PointerType, MetaData, true, FieldDimension>, Number1, Number2, Number3 >{
    //     typedef storage<data_field< storage_list<base_storage<PointerType, MetaData, true, Number1+Number2+Number3>, Number1-1>, storage_list<base_storage<PointerType, MetaData, true, Number1+Number2+Number3>, Number2-1>, storage_list<base_storage<PointerType, MetaData, true, Number1+Number2+Number3>, Number3-1> > > type;
    // };


    template<  typename PointerType
               ,typename MetaData
               ,short_t FieldDimension
               ,uint_t Number1, uint_t Number2, uint_t Number3 >
    struct field<no_storage_type_yet<storage<base_storage<PointerType, MetaData, FieldDimension> > >, Number1, Number2, Number3 >{
        typedef no_storage_type_yet<storage<data_field< storage_list<base_storage<PointerType, MetaData, Number1+Number2+Number3 >, Number1-1>, storage_list<base_storage<PointerType, MetaData, Number1+Number2+Number3 >, Number2-1>, storage_list<base_storage<PointerType, MetaData, Number1+Number2+Number3 >, Number3-1> > > > type;
    };
#endif

    template <typename T>
    std::ostream& operator<<(std::ostream &s, storage<T> const & x ) {
        s << "storage< "
          << static_cast<T const&>(x) << " > ";
        return s;
    }
}//namespace gridtools

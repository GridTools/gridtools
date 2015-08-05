#pragma once
#include "data_field.hpp"
#include "common/gpu_clone.hpp"
#include "host_tmp_storage.hpp"
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

      __device__
      storage(storage const& other)
          :  super(other)
      {}

        explicit storage(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3, value_type const& value, char const* s="default name"): super(dim1, dim2, dim3, value, s) {
            GRIDTOOLS_STATIC_ASSERT( boost::is_float<value_type>::value, "The initialization value in the storage constructor must me a floating point number (e.g. 1.0). \nIf you want to store an integer you have to split construction and initialization \n(using the member \"initialize\"). This because otherwise the initialization value would be interpreted as an extra dimension");
        }


        explicit storage(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3, value_type* ptr, char const* s="default name"): super(dim1, dim2, dim3, ptr, s) {}

#if defined(CXX11_ENABLED)
//arbitrary dimensional field
        template <class ... UIntTypes>
        explicit storage(  UIntTypes const& ... args/*, value_type init, char const* s*/ ):super(args ...)
            {
            }
#else
        //constructor picked in absence of CXX11 or which GCC<4.9
        explicit storage(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3): super(dim1, dim2, dim3) {}
#endif

//    private :
        explicit storage():super(){}
    };

    /**@brief Convenient syntactic sugar for specifying an extended-dimension with extended-width storages, where each dimension has arbitrary size 'Number'.

       Annoyngly enough does not work with CUDA 6.5
    */
#if defined(CXX11_ENABLED) && !defined(__CUDACC__)

    template< class Storage, uint_t ... Number >
    struct field_reversed{
        typedef storage< data_field< storage_list<base_storage<typename Storage::pointer_type, typename  Storage::layout, Storage::is_temporary, accumulate(add_functor(), ((uint_t)Number) ... )>, Number-1> ... > > type;
    };

    // template< class TmpStorage, uint_t ... Number >
    // struct tmp_field;

    template < typename PointerType
               , typename Layout
               , short_t FieldDimension
               , uint_t TileI
               , uint_t TileJ
               , uint_t MinusI
               , uint_t MinusJ
               , uint_t PlusI
               , uint_t PlusJ
               , uint_t ... Number >
    struct field_reversed<host_tmp_storage<base_storage< PointerType, Layout , true, FieldDimension>, TileI, TileJ, MinusI, MinusJ, PlusI, PlusJ>, Number... >{
        typedef storage<host_tmp_storage<data_field< storage_list<base_storage<PointerType, Layout, true, accumulate(add_functor(), ((uint_t)Number) ... )> , Number-1> ... >, TileI, TileJ, MinusI, MinusJ, PlusI, PlusJ> > type;
    };

    template<  typename PointerType
               ,typename Layout
               ,short_t FieldDimension
               ,uint_t ... Number >
    struct field_reversed<base_storage<PointerType, Layout, true, FieldDimension>, Number... >{
        typedef storage< data_field< storage_list<base_storage<PointerType, Layout, true, accumulate(add_functor(), ((uint_t)Number) ... )>, Number-1> ... > > type;
    };


    template<  typename PointerType
               ,typename Layout
               ,short_t FieldDimension
               ,uint_t ... Number >
    struct field_reversed<no_storage_type_yet<storage<base_storage<PointerType, Layout, true, FieldDimension> > >, Number... >{
        typedef no_storage_type_yet<storage<data_field< storage_list<base_storage<PointerType, Layout, true, accumulate(add_functor(), ((uint_t)Number) ... ) >, Number-1> ... > > > type;
    };

    template< class Storage, uint_t First, uint_t ... Number >
    struct field{
        typedef typename reverse_pack<Number ...>::template apply<field_reversed, Storage, First >::type::type type;
    };

#else//CXX11_ENABLED


    template< class Storage, uint_t Number1, uint_t Number2, uint_t Number3 >
    struct field{
        typedef storage< data_field< storage_list<base_storage<typename Storage::pointer_type, typename  Storage::layout, Storage::is_temporary, Number1+Number2+Number3>, Number1-1>, storage_list<base_storage<typename Storage::pointer_type, typename  Storage::layout, Storage::is_temporary, Number1+Number2+Number3>, Number2-1>, storage_list<base_storage<typename Storage::pointer_type, typename  Storage::layout, Storage::is_temporary, Number1+Number2+Number3>, Number3-1> > > type;
    };


    template< class Storage, uint_t Number1>
    struct field1{
        typedef storage< data_field1< storage_list<base_storage<typename Storage::pointer_type, typename  Storage::layout, Storage::is_temporary, Number1>, Number1-1> > > type;
    };



    // template< class TmpStorage, uint_t ... Number >
    // struct tmp_field;

    template <  typename PointerType
               , typename Layout
               , short_t FieldDimension
               , uint_t TileI
               , uint_t TileJ
               , uint_t MinusI
               , uint_t MinusJ
               , uint_t PlusI
               , uint_t PlusJ
               , uint_t Number1, uint_t Number2, uint_t Number3 >
    struct field<host_tmp_storage<base_storage<PointerType, Layout, true, FieldDimension>, TileI, TileJ, MinusI, MinusJ, PlusI, PlusJ>, Number1, Number2, Number3 >{
        typedef storage< host_tmp_storage<data_field< storage_list<base_storage<PointerType, Layout, true, Number1+Number2+Number3>, Number1-1>, storage_list<base_storage<PointerType, Layout, true, Number1+Number2+Number3>, Number2-1>, storage_list<base_storage<PointerType, Layout, true, Number1+Number2+Number3>, Number3-1> >, TileI, TileJ, MinusI, MinusJ, PlusI, PlusJ> > type;
    };

    template<  typename PointerType
               ,typename Layout
               ,short_t FieldDimension
               , uint_t Number1, uint_t Number2, uint_t Number3 >
    struct field<base_storage<PointerType, Layout, true, FieldDimension>, Number1, Number2, Number3 >{
        typedef storage<data_field< storage_list<base_storage<PointerType, Layout, true, Number1+Number2+Number3>, Number1-1>, storage_list<base_storage<PointerType, Layout, true, Number1+Number2+Number3>, Number2-1>, storage_list<base_storage<PointerType, Layout, true, Number1+Number2+Number3>, Number3-1> > > type;
    };


    template<  typename PointerType
               ,typename Layout
               ,short_t FieldDimension
               ,uint_t Number1, uint_t Number2, uint_t Number3 >
    struct field<no_storage_type_yet<storage<base_storage<PointerType, Layout, true, FieldDimension> > >, Number1, Number2, Number3 >{
        typedef no_storage_type_yet<storage<data_field< storage_list<base_storage<PointerType, Layout, true, Number1+Number2+Number3 >, Number1-1>, storage_list<base_storage<PointerType, Layout, true, Number1+Number2+Number3 >, Number2-1>, storage_list<base_storage<PointerType, Layout, true, Number1+Number2+Number3 >, Number3-1> > > > type;
    };
#endif

    template <typename T>
    std::ostream& operator<<(std::ostream &s, storage<T> const & x ) {
        s << "storage< "
          << static_cast<T const&>(x) << " > ";
        return s;
    }
}//namespace gridtools

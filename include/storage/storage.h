#include "base_storage.h"
#include "host_tmp_storage.h"

namespace gridtools{
/**@brief Convenient syntactic sugar for specifying an extended-dimension with extended-width storages, where each dimension has arbitrary size 'Number'.

       Annoyngly enough does not work with CUDA 6.5
    */
#if !defined(__CUDACC__)
    template< class Storage, uint_t ... Number >
    struct field{
	typedef extend_dim< extend_width<base_storage<Storage::backend, typename Storage::value_type, typename  Storage::layout, Storage::is_temporary, accumulate(add(), ((uint_t)Number) ... )>, Number-1> ... > type;
    };

    // template< class TmpStorage, uint_t ... Number >
    // struct tmp_field;

    template < enumtype::backend Backend
               , typename ValueType
               , typename Layout
               , short_t FieldDimension
               , uint_t TileI
               , uint_t TileJ
               , uint_t MinusI
               , uint_t MinusJ
               , uint_t PlusI
               , uint_t PlusJ
               , uint_t ... Number >
    struct field<host_tmp_storage<base_storage<Backend, ValueType, Layout, FieldDimension>, TileI, TileJ, MinusI, MinusJ, PlusI, PlusJ>, Number... >{
	typedef host_tmp_storage<extend_dim< extend_width<base_storage<Backend, ValueType, Layout, true, accumulate(add(), ((uint_t)Number) ... )> , Number-1> ... >, TileI, TileJ, MinusI, MinusJ, PlusI, PlusJ> type;
    };

    template<  enumtype::backend Backend
               ,typename ValueType
               ,typename Layout
	       ,short_t FieldDimension
               ,uint_t ... Number >
    struct field<base_storage<Backend, ValueType, Layout, true, FieldDimension>, Number... >{
	typedef extend_dim< extend_width<base_storage<Backend, ValueType, Layout, true, accumulate(add(), ((uint_t)Number) ... )>, Number-1> ... > type;
    };


    template<  enumtype::backend Backend
               ,typename ValueType
               ,typename Layout
	       ,short_t FieldDimension
               ,uint_t ... Number >
    struct field<no_storage_type_yet<base_storage<Backend, ValueType, Layout, true, FieldDimension> >, Number... >{
	typedef no_storage_type_yet<extend_dim< extend_width<base_storage<Backend, ValueType, Layout, true, accumulate(add(), ((uint_t)Number) ... ) >, Number-1> ... > > type;
    };

    // template<  enumtype::backend Backend
    //            ,typename ValueType
    //            ,typename Layout
    //            ,short_t FieldDimension
    //            ,uint_t ... Number >
    // struct field<no_storage_type_yet<base_storage<Backend, ValueType, Layout, true, FieldDimension> >, Number... >{
    //     typedef extend_dim< extend_width<host_tmp_storage<Backend, ValueType, Layout, accumulate(add(), ((uint_t)Number) ... ), 2, 2, 0, 0, 0, 0 >, Number-1> ... > type;
    // };


#endif

}//namespace gridtools

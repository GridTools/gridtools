#pragma once
#include<common/defs.h>
#include"base_storage.h"

/**
@file
@brief Storage class
This extra layer is added on top of the base_storage class because it extends the clonabl_to_gpu interface. Due to the multiple inheritance pattern this class should not be further inherited.
*/

namespace gridtools {
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

	template<typename FloatType>
        explicit storage(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3 ,
                         FloatType init = float(), char const* s = "default name" ): super(dim1, dim2, dim3, init, s) {
        }


	template<typename FloatType>
	explicit storage(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3, typename BaseStorage::value_type* ptr, char const* s="default name"): super(dim1, dim2, dim3, ptr, s) {}

#ifdef CXX11_ENABLED
#if !defined(__GNUC__) || (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 9) )
	//arbitrary dimensional field
	template <class ... UIntTypes>
	explicit storage(  UIntTypes const& ... args/*, value_type init, char const* s*/ ):super(args ...)
            {
	    }
#endif
#endif
	//    private :
	explicit storage():super(){}
    };


}//namespace gridtools

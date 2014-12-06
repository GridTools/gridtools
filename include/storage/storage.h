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

        explicit storage(uint_t dim1, uint_t dim2, uint_t dim3,
                         typename BaseStorage::value_type init = typename BaseStorage::value_type(), std::string const& s = std::string("default name") ): super(dim1, dim2, dim3, init, s) {
        }

	explicit storage(uint_t dim1, uint_t dim2, uint_t dim3, typename BaseStorage::value_type* ptr,
			      typename BaseStorage::value_type init = value_type(), std::string const& s = std::string("default name") ): super(dim1, dim2, dim3, ptr, s) {}

	//    private :
	explicit storage():super(){}
    };


}//namespace gridtools

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

	explicit storage(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3, value_type const& value, char const* s="default name"): super(dim1, dim2, dim3, value, s) {
            GRIDTOOLS_STATIC_ASSERT( boost::is_float<value_type>::value, "The initialization value in the storage constructor must me a floating point number (e.g. 1.0). \nIf you want to store an integer you have to split construction and initialization \n(using the member \"initialize\"). This because otherwise the initialization value would be interpreted as an extra dimension");
        }


	explicit storage(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3, value_type* ptr, char const* s="default name"): super(dim1, dim2, dim3, ptr, s) {}

#if defined(CXX11_ENABLED) && ( !defined(__GNUC__) || (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 9) ) )
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

    template <typename T>
    std::ostream& operator<<(std::ostream &s, storage<T> const & x ) {
        s << "storage< "
          << static_cast<T const&>(x) << " > ";
        return s;
    }

}//namespace gridtools

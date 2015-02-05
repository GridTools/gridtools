#pragma once
#include<common/defs.h>

/**
@file
@brief Parallel Storage class
This file defines a policy class extending the storage to distributed memory.
*/

namespace gridtools {

    template < typename SerialStorage >
    class parallel_storage : public SerialStorage
    {

    public:
	typedef SerialStorage super;
	typedef typename SerialStorage::basic_type basic_type;
	typedef parallel_storage<SerialStorage> original_storage;
	typedef clonable_to_gpu<parallel_storage<SerialStorage> > gpu_clone;
	typedef typename SerialStorage::iterator_type iterator_type;
	typedef typename SerialStorage::value_type value_type;
	static const ushort_t n_args = basic_type::n_width;

        __device__
	parallel_storage(parallel_storage const& other)
            :  super(other)
            {}

        //3D
        explicit parallel_storage(partitioner& part, uint_t const& d1, uint_t const& d2, uint_t const& d3)
            : super(part.compute_bounds(0, d1), part.compute_bounds(1, d2), part.compute_bounds(2, d3))
            , m_partitioner(&part)
            {
            }

    private:
        partitioner* m_partitioner;
    };
}//namespace gridtools

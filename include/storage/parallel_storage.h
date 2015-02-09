#pragma once
#include<common/defs.h>
#include "partitioner.h"
/**
@file
@brief Parallel Storage class
This file defines a policy class extending the storage to distributed memory.
*/

namespace gridtools {

    template < typename Partitioner >
    class parallel_storage : public Partitioner::storage_t
    {

    public:
        typedef Partitioner partitioner_t;
	typedef typename partitioner_t::storage_t super;
	typedef typename partitioner_t::storage_t::basic_type basic_type;
	typedef parallel_storage<partitioner_t> original_storage;
	typedef clonable_to_gpu<parallel_storage<partitioner_t> > gpu_clone;
	typedef typename partitioner_t::storage_t::iterator_type iterator_type;
	typedef typename partitioner_t::storage_t::value_type value_type;
	static const ushort_t n_args = basic_type::n_width;

        __device__
	parallel_storage(parallel_storage const& other)
            :  super(other)
            {}

        //3D
        explicit parallel_storage(partitioner_t& part, uint_t const& d1, uint_t const& d2, uint_t const& d3)
            : super(part.compute_bounds(0, d1), part.compute_bounds(1, d2), part.compute_bounds(2, d3))
            , m_partitioner(&part)
            {
            }

    private:
        partitioner_t* m_partitioner;
    };
}//namespace gridtools

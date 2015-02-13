#pragma once
#include <common/defs.h>
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

        /**
           @brief 3D constructor

           Given the partitioner and the three space dimensions it constructs a storage and allocate the data
           relative to the current process
         */
        explicit parallel_storage(partitioner_t& part, uint_t const& d1, uint_t const& d2, uint_t const& d3)
            : super(part.compute_bounds(0, d1), part.compute_bounds(1, d2), part.compute_bounds(2, d3))
            , m_partitioner(&part)
            {
            }

#ifdef CXX11_ENABLED
        template <uint_t field_dim=0, uint_t snapshot=0, typename ... UInt>
        typename super::value_type& get_value( UInt const& ... i )
		{
                    if(m_partitioner->mine(i...))
                        return super::template get<field_dim, snapshot>()[super::_index(i...)];
                    else
#ifndef DNDEBUG
                        printf("(%d, %d, %d) not available in processor %d \n\n", i ... , m_partitioner->template pid<0>()+m_partitioner->template pid<1>()+m_partitioner->template pid<2>());
#endif
                    return -1.;
		}
#endif

        /**
         @brief given a local (to the current subdomain) index (i,j,k) it returns the global corresponding index

         It sums the offset given by the partitioner to the local index
        */
        template<uint_t Component>
        uint_t const& local_to_global(uint_t const& value){return m_partitioner->template global_offset<Component>()+value;}

    private:
        partitioner_t* m_partitioner;
    };
}//namespace gridtools

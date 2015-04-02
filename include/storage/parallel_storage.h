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
        tatic const ushort_t n_args = basic_type::n_width;

        __device__
        parallel_storage(parallel_storage const& other)
            :  super(other)
            {}

        explicit parallel_storage(partitioner_t const& part)
            : super()
            , m_partitioner(&part)
            {
            }

        /**
           @brief 3D constructor

           Given the partitioner and the three space dimensions it constructs a storage and allocate the data
           relative to the current process
         */
        void setup(uint_t const& d1, uint_t const& d2, uint_t const& d3)
            {
                uint_t dims[3];
                m_partitioner->compute_bounds(dims, m_coordinates, m_coordinates_gcl, m_low_bound, m_up_bound,  d1, d2, d3);
                super::setup(dims[0], dims[1], dims[2]);
            }

#ifdef CXX11_ENABLED

        // template <typename ... UInt >
        // explicit parallel_storage(partitioner_t& part, UInt const& ... di)
        //     : super(part.compute_bounds(0, m_coordinates, m_coordinates_gcl, m_low_bound, m_up_bound, di ... ) )
        //     , m_partitioner(&part)
        //     {
        //     }

        template <uint_t field_dim=0, uint_t snapshot=0, typename ... UInt>
        typename super::value_type& get_value( UInt const& ... i )
            {
                    if(m_partitioner->mine(i...))
                        return super::template get<field_dim, snapshot>()[super::_index(super::strides(), i...)];
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

        template<ushort_t dimension>
        halo_descriptor const& get_halo_descriptor() const {return m_coordinates[dimension];}

        template<ushort_t dimension>
        halo_descriptor const& get_halo_gcl() const {return m_coordinates_gcl[dimension];}

    private:

        partitioner_t const* m_partitioner;
        //these are set by the partitioner
        halo_descriptor m_coordinates[super::space_dimensions];
        halo_descriptor m_coordinates_gcl[super::space_dimensions];
        //these remember where am I on the storage (also set by the partitioner)
        int_t m_low_bound[super::space_dimensions];
        int_t m_up_bound[super::space_dimensions];
    };
}//namespace gridtools

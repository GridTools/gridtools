#pragma once
#include "common/defs.hpp"
#include "partitioner.hpp"
/**
   @file
   @brief Parallel Storage class
   This file defines a policy class extending the storage to distributed memory.
*/

namespace gridtools {

    template < typename MetaStorage, typename Partitioner >
    class parallel_meta_storage
    {

    public:
        typedef Partitioner partitioner_t;
        typedef MetaStorage metadata_t;

    private:
        partitioner_t const* m_partitioner;
        //these are set by the partitioner
        array<halo_descriptor, metadata_t::space_dimensions> m_coordinates;
        array<halo_descriptor, metadata_t::space_dimensions> m_coordinates_gcl;
        //these remember where am I on the storage (also set by the partitioner)
        array<int_t, metadata_t::space_dimensions> m_low_bound;
        array<int_t, metadata_t::space_dimensions> m_up_bound;
        metadata_t m_metadata;

    public:
        DISALLOW_COPY_AND_ASSIGN(parallel_meta_storage);

        explicit parallel_meta_storage(partitioner_t const& part)
            : m_partitioner(&part)
            , m_metadata()
            {
            }

        template <typename ... UInt>
        explicit parallel_meta_storage(partitioner_t const& part, UInt const& ... components_)
            : m_partitioner(&part)
            , m_metadata(part.compute_bounds(components_, m_coordinates, m_coordinates_gcl, m_low_bound, m_up_bound)...)
            {
            }

        /**
           @brief 3D constructor

           Given the partitioner and the three space dimensions it constructs a storage and allocate the data
           relative to the current process
        // */
        // void setup(uint_t const& d1, uint_t const& d2, uint_t const& d3)
        //     {
        //         /**the partitioner must be set at this point*/
        //         assert(m_partitioner);
        //         uint_t dims[3];
        //         m_partitioner->compute_bounds(dims, m_coordinates, m_coordinates_gcl, m_low_bound, m_up_bound,  d1, d2, d3);
        //         m_metadata=MetaStorage(dims[0], dims[1], dims[2]);
        //     }

#ifdef CXX11_ENABLED

        template <typename ... UInt>
        bool mine(UInt const& ... coordinates_)
            {
                GRIDTOOLS_STATIC_ASSERT((sizeof ... (UInt) >= metadata_t::space_dimensions), "not enough indices specified in the call to parallel_meta_storage::mine()");
                GRIDTOOLS_STATIC_ASSERT((sizeof ... (UInt) <= metadata_t::space_dimensions), "too many indices specified in the call to parallel_meta_storage::mine()");
                uint_t coords[metadata_t::space_dimensions]={coordinates_ ...};
                bool result=true;
                for(ushort_t i=0; i<metadata_t::space_dimensions; ++i)
                    if(coords[i]<m_low_bound[i]+m_coordinates[i].begin() || coords[i]>m_low_bound[i]+m_coordinates[i].end() )
                        result=false;
                return result;
            }

        //TODO generalize for arbitrary dimension
        template <uint_t field_dim=0, uint_t snapshot=0, typename UInt>
        uint_t get_local_index( UInt const& i, UInt const& j, UInt const& k )
            {
                if(mine(i,j,k))
                    return m_metadata._index(i-m_low_bound[0], j-m_low_bound[1], k-m_low_bound[2]);
                else
#ifndef DNDEBUG
                    printf("(%d, %d, %d) not available in processor %d \n\n", i, j, k , m_partitioner->template pid<0>()+m_partitioner->template pid<1>()+m_partitioner->template pid<2>());
#endif
                return -1.;
            }
#endif

        /**
           @brief given a local (to the current subdomain) index (i,j,k) it returns the global corresponding index

           It sums the offset given by the partitioner to the local index
        */
        template<uint_t Component>
        uint_t const& local_to_global(uint_t const& value){
            GRIDTOOLS_STATIC_ASSERT(Component<metadata_t::space_dimensions, "only positive integers smaller than the number of dimensions are accepted as template arguments of local_to_global");
            return m_partitioner->template global_offset<Component>()+value;}

        /**
           @brief returns the halo descriptors to be used inside the coordinates object

           The halo descriptors are computed in the setup phase using the partitioner
        */
        template<ushort_t dimension>
        halo_descriptor const& get_halo_descriptor() const {
            GRIDTOOLS_STATIC_ASSERT(dimension<metadata_t::space_dimensions, "only positive integers smaller than the number of dimensions are accepted as template arguments of get_halo_descriptor");
            return m_coordinates[dimension];}

        /**
           @brief returns the halo descriptors to be used for the communication inside the GCL library

           The halo descriptors are computed in the setup phase using the partitioner
        */
        template<ushort_t dimension>
        halo_descriptor const& get_halo_gcl() const {return m_coordinates_gcl[dimension];}


        /**
           @brief returns the metadata with the meta information about the partitioned storage
        */
        metadata_t const& get_metadata() const {return m_metadata;}

    private:

        parallel_meta_storage();

    };
}//namespace gridtools

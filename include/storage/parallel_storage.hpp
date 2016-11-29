/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once
#include "common/defs.hpp"
#include "partitioner.hpp"
#include <functional>
/**
   @file
   @brief Parallel Storage class
   This file defines a policy class extending the storage to distributed memory.
*/

namespace gridtools {

    template < typename MetaStorage, typename Partitioner >
    class parallel_storage_info {

      public:
        typedef Partitioner partitioner_t;
        typedef MetaStorage metadata_t;

      private:
        partitioner_t const *m_partitioner;
        // these are set by the partitioner
        array< halo_descriptor, metadata_t::space_dimensions > m_coordinates;
        array< halo_descriptor, metadata_t::space_dimensions > m_coordinates_gcl;
        // these remember where am I in the storage (also set by the partitioner)
        array< int_t, metadata_t::space_dimensions > m_low_bound;
        array< int_t, metadata_t::space_dimensions > m_up_bound;
        metadata_t m_metadata;

      public:
        DISALLOW_COPY_AND_ASSIGN(parallel_storage_info);

        explicit parallel_storage_info(partitioner_t const &part) : m_partitioner(&part), m_metadata() {}

#ifdef CXX11_ENABLED

#ifndef __CUDACC__

        /**
           @brief constructor for the parallel meta storage

           \param part the partitioner (e.g. \ref gridtools::partitioner_trivial)
           \param dims_ ... the global dimensions of the storage

           The global dimensions get transformed to local ones during the construction
           NOTE: very convoluted way to initialize the metadata. it is initialized performing the
           following unroll operation
           \code
           m_metadata(
           part.compute_bounds(0, .....),
           part.compute_bounds(1, .....),
           part.compute_bounds(2, .....),
           part.compute_bounds(3, .....),
           .....)
           \endcode
           Each call to compute_bounds returns the partitioned dimension and fills an element
           of the arrays passed as input (corresponding to the dimension passed as first argument).

         */
        template < typename... UInt >
        explicit parallel_storage_info(partitioner_t const &part, UInt const &... dims_)
            : m_partitioner(&part), m_coordinates(), m_coordinates_gcl(), m_low_bound(), m_up_bound(),
              m_metadata(
                  apply_gt_integer_sequence< typename make_gt_integer_sequence< uint_t, sizeof...(UInt) >::type >::
                      template apply< metadata_t >(
                          ([&part](uint_t index_,
                               array< halo_descriptor, metadata_t::space_dimensions > & coordinates_,
                               array< halo_descriptor, metadata_t::space_dimensions > & coordinates_gcl_,
                               array< int_t, metadata_t::space_dimensions > & low_bound_,
                               array< int_t, metadata_t::space_dimensions > & up_bound_,
                               UInt const &... args_) -> uint_t {
                              return part.compute_bounds(
                                  index_, coordinates_, coordinates_gcl_, low_bound_, up_bound_, args_...);
                          }),
                          m_coordinates,
                          m_coordinates_gcl,
                          m_low_bound,
                          m_up_bound,
                          dims_...)) {}

#else

        template < typename... UInt >
        explicit parallel_storage_info(partitioner_t const &part, UInt const &... dims_)
            : m_partitioner(&part), m_coordinates(), m_coordinates_gcl(), m_low_bound(), m_up_bound(), m_metadata() {
            auto d1 = part.compute_bounds(0, m_coordinates, m_coordinates, m_low_bound, m_up_bound, dims_...);
            auto d2 = part.compute_bounds(1, m_coordinates, m_coordinates_gcl, m_low_bound, m_up_bound, dims_...);
            auto d3 = part.compute_bounds(2, m_coordinates, m_coordinates_gcl, m_low_bound, m_up_bound, dims_...);
            m_metadata.setup(d1, d2, d3);
        }

#endif

        /**
           @brief given the global grid returns wether the point belongs to the current partition
        */
        template < typename... UInt >
        bool mine(UInt const &... coordinates_) const {
            GRIDTOOLS_STATIC_ASSERT((sizeof...(UInt) >= metadata_t::space_dimensions),
                "not enough indices specified in the call to parallel_storage_info::mine()");
            GRIDTOOLS_STATIC_ASSERT((sizeof...(UInt) <= metadata_t::space_dimensions),
                "too many indices specified in the call to parallel_storage_info::mine()");
            uint_t coords[metadata_t::space_dimensions] = {coordinates_...};
            bool result = true;
            for (ushort_t i = 0; i < metadata_t::space_dimensions; ++i)
                if (coords[i] < m_low_bound[i] + m_coordinates[i].begin() ||
                    coords[i] > m_low_bound[i] + m_coordinates[i].end())
                    result = false;
            return result;
        }

        /**
           @brief given the global coordinates returns the local index, or -1 if the point is outside the current
           partition
        */
        // TODO generalize for arbitrary dimension
        template < uint_t field_dim = 0, uint_t snapshot = 0, typename UInt >
        int_t get_local_index(UInt const &i, UInt const &j, UInt const &k) const {
            if (mine(i, j, k))
                return m_metadata.index(
                    (uint_t)(i - m_low_bound[0]), (uint_t)(j - m_low_bound[1]), (uint_t)(k - m_low_bound[2]));
            else
#ifndef DNDEBUG
                printf("(%d, %d, %d) not available in processor %d \n\n",
                    i,
                    j,
                    k,
                    m_partitioner->template pid< 0 >() + m_partitioner->template pid< 1 >() +
                        m_partitioner->template pid< 2 >());
#endif
            return -1;
        }
#endif

        /**
           @brief given a local (to the current subdomain) index (i,j,k) it returns the global corresponding index

           It sums the offsets computed by the partitioner to the local index
        */
        template < uint_t Component >
        uint_t const &local_to_global(uint_t const &value) {
            GRIDTOOLS_STATIC_ASSERT(Component < metadata_t::space_dimensions,
                "only positive integers smaller than the "
                "number of dimensions are accepted as "
                "template arguments of local_to_global");
            return m_low_bound[Component] + value;
        }

        /**
           @brief returns the halo descriptors to be used inside the coordinates object

           The halo descriptors are computed in the setup phase using the partitioner
        */
        template < ushort_t dimension >
        halo_descriptor const &get_halo_descriptor() const {
            GRIDTOOLS_STATIC_ASSERT(dimension < metadata_t::space_dimensions,
                "only positive integers smaller than the number of dimensions are accepted as template arguments of "
                "get_halo_descriptor");
            return m_coordinates[dimension];
        }

        /**
           @brief returns the halo descriptors to be used for the communication inside the GCL library

           The halo descriptors are computed in the setup phase using the partitioner
        */
        template < ushort_t dimension >
        halo_descriptor const &get_halo_gcl() const {
            return m_coordinates_gcl[dimension];
        }

        /**
           @brief returns the metadata with the meta information about the partitioned storage
        */
        metadata_t const &get_metadata() const { return m_metadata; }

      private:
        parallel_storage_info();
    };
} // namespace gridtools

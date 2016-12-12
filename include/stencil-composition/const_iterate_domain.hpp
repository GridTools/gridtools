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
/**@file
   @brief file with classes to store the data members of the iterate domain
   that will be allocated in const memory
 */
#pragma once
#include <boost/mpl/has_key.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/at.hpp>
#include <boost/fusion/sequence/intrinsic/at_key.hpp>
#include <boost/fusion/include/at_key.hpp>
#include <boost/fusion/include/at.hpp>
#include "stencil-composition/accessor.hpp"
#include "common/generic_metafunctions/fusion_map_to_mpl_map.hpp"

namespace gridtools {

    /**
     * @class const_iterate_domain
     * data structure that holds data members of the iterate domain that must be stored in const memory.
     * @tparam DataPointerArray array of data pointers
     * @tparam StridesType strides cached type
     * @tparam IJCachesTuple fusion map of <index_type, cache_storage>
     */
    template < typename DataPointerArray, typename Strides, typename Dims, typename PEBlock, typename Backend >
    struct const_iterate_domain {

        typedef PEBlock processing_elements_block_size_t;
        typedef DataPointerArray data_pointer_array_t;
        typedef Strides strides_t;
        typedef Dims dims_t;

        GRIDTOOLS_STATIC_ASSERT((is_array_tuple< strides_t >::value), "Internal Error: wrong type");

      private:
        data_pointer_array_t m_data_pointer;
        strides_t m_strides;
        dims_t m_dims;

      public:
        const_iterate_domain() {}

        template < typename F, typename U, typename C, bool K >
        const_iterate_domain(local_domain< F, U, C, K > const &local_domain_, int_t EU_id_i, int_t EU_id_j) {
            assign_storage_pointers(local_domain_, EU_id_i, EU_id_j);
            assign_stride_pointers(local_domain_);
            assign_dims_pointers(local_domain_);
        }

        GT_FUNCTION
        const_iterate_domain(const_iterate_domain const &other_)
            : m_data_pointer(other_.m_data_pointer), m_strides(other_.m_strides), m_dims(other_.m_dims) {}

        GT_FUNCTION
        data_pointer_array_t const &data_pointer() const { return m_data_pointer; }
        GT_FUNCTION
        strides_t const &strides() const { return m_strides; }
        GT_FUNCTION
        dims_t const &dims() const { return m_dims; }
        GT_FUNCTION
        data_pointer_array_t &data_pointer() { return m_data_pointer; }

      private:
        GT_FUNCTION
        dims_t &dims() { return m_dims; }
        GT_FUNCTION
        strides_t &strides() { return m_strides; }

        /** This functon set the addresses of the data values  before the computation
            begins.

            The EU stands for ExecutionUnit (thich may be a thread or a group of
            threasd. There are potentially two ids, one over i and one over j, since
            our execution model is parallel on (i,j). Defaulted to 1.
        */
        template < typename LocalDomain >
        GT_FUNCTION void assign_storage_pointers(LocalDomain const &local_domain_, int_t EU_id_i, int_t EU_id_j) {
            typedef LocalDomain local_domain_t;

            boost::mpl::for_each< typename boost::mpl::
                    range_c< uint_t, 0, boost::mpl::size< typename local_domain_t::actual_args_type >::value >::type >(
                assign_storage_functor< Backend,
                    data_pointer_array_t,
                    typename local_domain_t::local_storage_type,
                    typename local_domain_t::local_metadata_type,
                    typename local_domain_t::storage_metadata_map,
                    processing_elements_block_size_t >(
                    data_pointer(), local_domain_.local_storages(), local_domain_.local_metadata(), EU_id_i, EU_id_j));
        }

        /**
           @brief recursively assignes all the strides

           copies them from the
           local_domain.m_local_metadata vector, and stores them into an instance of the
           \ref array_tuple class.
         */
        template < typename LocalDomain >
        GT_FUNCTION void assign_stride_pointers(LocalDomain const &local_domain_) {
            typedef LocalDomain local_domain_t;
            GRIDTOOLS_STATIC_ASSERT((is_array_tuple< strides_t >::value), "internal error type");
            boost::mpl::for_each< typename local_domain_t::storage_metadata_map >(assign_strides_functor< strides_t,
                typename boost::fusion::result_of::as_vector< typename local_domain_t::local_metadata_type >::type >(
                strides(), local_domain_.m_local_metadata));
        }

        /**
           @brief recursively assignes all the dims

           copies them from the
           local_domain.m_local_metadata vector, and stores them into an instance of the
           \ref array_tuple class.
         */
        template < typename LocalDomain >
        GT_FUNCTION void assign_dims_pointers(LocalDomain const &local_domain_) {
            typedef LocalDomain local_domain_t;
            GRIDTOOLS_STATIC_ASSERT((is_array_tuple< strides_t >::value), "internal error type");
            boost::mpl::for_each< typename local_domain_t::storage_metadata_map >(assign_dims_functor< dims_t,
                typename boost::fusion::result_of::as_vector< typename local_domain_t::local_metadata_type >::type >(
                m_dims, local_domain_.m_local_metadata));
        }
    };

} // namespace gridtools

/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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

#include "stencil-composition/iterate_domain.hpp"
#include "stencil-composition/iterate_domain_fwd.hpp"
#include "stencil-composition/iterate_domain_impl_metafunctions.hpp"
#include "stencil-composition/iterate_domain_metafunctions.hpp"

namespace gridtools {

    /**
 * @brief iterate domain class for the Host backend
 */
    template < template < class > class IterateDomainBase, typename IterateDomainArguments >
    class iterate_domain_host
        : public IterateDomainBase< iterate_domain_host< IterateDomainBase, IterateDomainArguments > > // CRTP
    {
        DISALLOW_COPY_AND_ASSIGN(iterate_domain_host);
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments< IterateDomainArguments >::value), GT_INTERNAL_ERROR);

        typedef IterateDomainBase< iterate_domain_host< IterateDomainBase, IterateDomainArguments > > super;

      public:
        typedef iterate_domain_host iterate_domain_t;
        typedef typename super::data_ptr_cached_t data_ptr_cached_t;
        typedef typename super::strides_cached_t strides_cached_t;
        typedef typename super::local_domain_t local_domain_t;
        typedef typename super::grid_topology_t grid_topology_t;
        typedef boost::mpl::map0<> ij_caches_map_t;

        GT_FUNCTION
        explicit iterate_domain_host(local_domain_t const &local_domain_, grid_topology_t const &grid_topology)
            : super(local_domain_, grid_topology), m_data_pointer(0), m_strides(0) {}

        void set_data_pointer_impl(data_ptr_cached_t *RESTRICT data_pointer) {
            assert(data_pointer);
            m_data_pointer = data_pointer;
        }

        data_ptr_cached_t &RESTRICT data_pointer_impl() {
            assert(m_data_pointer);
            return *m_data_pointer;
        }

        data_ptr_cached_t const &RESTRICT data_pointer_impl() const {
            assert(m_data_pointer);
            return *m_data_pointer;
        }

        strides_cached_t &RESTRICT strides_impl() {
            assert(m_strides);
            return *m_strides;
        }

        strides_cached_t const &RESTRICT strides_impl() const {
            assert(m_strides);
            return *m_strides;
        }

        void set_strides_pointer_impl(strides_cached_t *RESTRICT strides) {
            assert(strides);
            m_strides = strides;
        }

        template < ushort_t Coordinate, typename Execution >
        GT_FUNCTION void increment_impl() {}

        template < ushort_t Coordinate >
        GT_FUNCTION void increment_impl(int_t steps) {}

        template < ushort_t Coordinate >
        GT_FUNCTION void initialize_impl() {}

        template < typename ReturnType, typename Accessor, typename StoragePointer >
        GT_FUNCTION ReturnType get_value_impl(
            StoragePointer RESTRICT &storage_pointer, const uint_t pointer_offset) const {
            GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), GT_INTERNAL_ERROR);

            return super::template get_gmem_value< ReturnType >(storage_pointer, pointer_offset);
        }

        template < typename IterationPolicy >
        GT_FUNCTION void slide_caches() {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
        }
        template < typename IterationPolicy, typename Grid >
        GT_FUNCTION void flush_caches(const int_t klevel, Grid const &grid) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
            GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "error");
        }
        template < typename IterationPolicy, typename Grid >
        GT_FUNCTION void fill_caches(const int_t klevel, Grid const &grid) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
            GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "error");
        }
        template < typename IterationPolicy >
        GT_FUNCTION void final_flush() {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
        }
        template < typename IterationPolicy >
        GT_FUNCTION void begin_fill() {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
        }

        template < typename Extent >
        GT_FUNCTION bool is_thread_in_domain() const {
            return true;
        }

      private:
        data_ptr_cached_t *RESTRICT m_data_pointer;
        strides_cached_t *RESTRICT m_strides;
    };

    template < template < class > class IterateDomainBase, typename IterateDomainArguments >
    struct is_iterate_domain< iterate_domain_host< IterateDomainBase, IterateDomainArguments > >
        : public boost::mpl::true_ {};

    //    template<
    //            template<class> class IterateDomainBase,
    //            typename IterateDomainArguments
    //            >
    //    struct is_positional_iterate_domain<iterate_domain_host<IterateDomainBase, IterateDomainArguments> > :
    //            is_positional_iterate_domain<IterateDomainBase<iterate_domain_host<IterateDomainBase,
    //            IterateDomainArguments> > > {};

} // namespace gridtools

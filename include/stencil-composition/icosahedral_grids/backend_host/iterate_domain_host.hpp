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
        GRIDTOOLS_STATIC_ASSERT(
            (is_iterate_domain_arguments< IterateDomainArguments >::value), "Internal error: wrong type");

        typedef IterateDomainBase< iterate_domain_host< IterateDomainBase, IterateDomainArguments > > super;

      public:
        typedef iterate_domain_host iterate_domain_t;
        typedef typename super::data_pointer_array_t data_pointer_array_t;
        typedef typename super::strides_cached_t strides_cached_t;
        typedef typename super::local_domain_t local_domain_t;
        typedef typename super::grid_topology_t grid_topology_t;
        typedef boost::mpl::map0<> ij_caches_map_t;

        GT_FUNCTION
        explicit iterate_domain_host(local_domain_t const &local_domain_, grid_topology_t const &grid_topology)
            : super(local_domain_, grid_topology), m_data_pointer(0), m_strides(0) {}

        void set_data_pointer_impl(data_pointer_array_t *RESTRICT data_pointer) {
            assert(data_pointer);
            m_data_pointer = data_pointer;
        }

        data_pointer_array_t &RESTRICT data_pointer_impl() {
            assert(m_data_pointer);
            return *m_data_pointer;
        }
        data_pointer_array_t const &RESTRICT data_pointer_impl() const {
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

      private:
        data_pointer_array_t *RESTRICT m_data_pointer;
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

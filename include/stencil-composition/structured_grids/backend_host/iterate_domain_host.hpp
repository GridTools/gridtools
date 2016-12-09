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

#include "stencil-composition/iterate_domain_fwd.hpp"
#include "stencil-composition/iterate_domain.hpp"
#include "stencil-composition/iterate_domain_metafunctions.hpp"
#include "stencil-composition/iterate_domain_impl_metafunctions.hpp"
#include "../../const_iterate_domain.hpp"

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

        typedef typename IterateDomainArguments::local_domain_t local_domain_t;
        typedef typename super::reduction_type_t reduction_type_t;

      public:
        using super::operator();
        typedef iterate_domain_host iterate_domain_t;
        typedef typename super::data_pointer_array_t data_pointer_array_t;
        typedef typename super::array_tuple_t array_tuple_t;
        typedef typename super::dims_cached_t dims_cached_t;
        typedef boost::mpl::map0<> ij_caches_map_t;

        typedef const_iterate_domain< data_pointer_array_t,
            array_tuple_t,
            dims_cached_t,
            typename IterateDomainArguments::processing_elements_block_size_t,
            backend_traits_from_id< enumtype::Host > > const_iterate_domain_t;

      private:
        const_iterate_domain_t const *RESTRICT m_pconst_iterate_domain;

      public:
        GT_FUNCTION
        explicit iterate_domain_host(const reduction_type_t &reduction_initial_value)
            : super(reduction_initial_value) {}

        data_pointer_array_t const &RESTRICT data_pointer_impl() const {
            return m_pconst_iterate_domain->data_pointer();
        }

        array_tuple_t const &RESTRICT strides_impl() const { return m_pconst_iterate_domain->strides(); }

        GT_FUNCTION
        void set_const_iterate_domain_pointer_impl(const_iterate_domain_t const *ptr) { m_pconst_iterate_domain = ptr; }

        iterate_domain_host const &get() const { return *this; }

        template < ushort_t Coordinate, typename Execution >
        GT_FUNCTION void increment_impl() {}

        template < ushort_t Coordinate >
        GT_FUNCTION void increment_impl(int_t steps) {}

        template < ushort_t Coordinate >
        GT_FUNCTION void initialize_impl() {}

        template < typename ReturnType, typename Accessor, typename StoragePointer >
        GT_FUNCTION ReturnType get_value_impl(
            typename StoragePointer::value_type RESTRICT *storage_pointer, const uint_t pointer_offset) const {
            GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Wrong type");

            return super::template get_gmem_value< ReturnType >(storage_pointer, pointer_offset);
        }
    };

    template < template < class > class IterateDomainBase, typename IterateDomainArguments >
    struct is_iterate_domain< iterate_domain_host< IterateDomainBase, IterateDomainArguments > >
        : public boost::mpl::true_ {};

    template < template < class > class IterateDomainBase, typename IterateDomainArguments >
    struct is_positional_iterate_domain< iterate_domain_host< IterateDomainBase, IterateDomainArguments > >
        : is_positional_iterate_domain<
              IterateDomainBase< iterate_domain_host< IterateDomainBase, IterateDomainArguments > > > {};

} // namespace gridtools

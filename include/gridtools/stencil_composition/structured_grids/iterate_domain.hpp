/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/host_device.hpp"
#include "../../common/hymap.hpp"
#include "../../meta.hpp"
#include "../dim.hpp"
#include "../iterate_domain_aux.hpp"
#include "../local_domain.hpp"
#include "../pos3.hpp"
#include "../sid/multi_shift.hpp"

namespace gridtools {
    /**@brief class managing the memory accesses, indices increment

       This class gets instantiated in the backend-specific code, and has a different implementation for
       each backend (see CRTP pattern). It is instantiated within the kernel (e.g. in the device code),
       and drives all the operations which are performed at the innermost level. In particular
       the computation/increment of the useful addresses in memory, given the iteration point,
       the storage placeholders/metadatas and their offsets.
     */
    template <class IterateDomainArguments>
    class iterate_domain {
      private:
        using local_domain_t = typename IterateDomainArguments::local_domain_t;
        GT_STATIC_ASSERT(is_local_domain<local_domain_t>::value, GT_INTERNAL_ERROR);

        // the number of different storage metadatas used in the current functor
        static const uint_t n_meta_storages = meta::length<typename local_domain_t::strides_kinds_t>::value;

      public:
        using array_index_t = array<int_t, n_meta_storages>;

      protected:
        using iterate_domain_arguments_t = IterateDomainArguments;

        local_domain_t const &m_local_domain;

        typename local_domain_t::ptr_map_t m_ptr_map;
        array_index_t m_index;

        template <class Dim, class Offset>
        GT_FUNCTION void increment(Offset const &offset) {
            do_increment<Dim, local_domain_t>(offset, m_local_domain.m_strides_map, m_index);
        }

        template <class Arg, class Accessor>
        GT_FUNCTION auto get_ptr(Accessor const &acc) const -> decay_t<decltype(host_device::at_key<Arg>(m_ptr_map))> {
            using storage_info_t = typename Arg::data_store_t::storage_info_t;

            static constexpr auto storage_info_index =
                meta::st_position<typename local_domain_t::strides_kinds_t, storage_info_t>::value;

            auto offset = m_index[storage_info_index];
            sid::multi_shift(offset, host_device::at_key<storage_info_t>(m_local_domain.m_strides_map), acc);

            return host_device::at_key<Arg>(m_ptr_map) + offset;
        }

      public:
        static constexpr bool has_k_caches = false;

        GT_FUNCTION_DEVICE iterate_domain(local_domain_t const &local_domain_)
            : m_local_domain(local_domain_), m_ptr_map(local_domain_.make_ptr_map()) {}

        GT_FUNCTION array_index_t const &index() const { return m_index; }

        /**@brief method for setting the index array
         * This method is responsible of assigning the index for the memory access at
         * the location (i,j,k). Such index is shared among all the fields contained in the
         * same storage class instance, and it is not shared among different storage instances.
         */
        GT_FUNCTION void set_index(array_index_t const &index) { m_index = index; }

        template <class Offset = integral_constant<int_t, 1>>
        GT_FUNCTION void increment_i(Offset const &offset = {}) {
            increment<dim::i>(offset);
        }
        template <class Offset = integral_constant<int_t, 1>>
        GT_FUNCTION void increment_j(Offset const &offset = {}) {
            increment<dim::j>(offset);
        }
        template <class Offset = integral_constant<int_t, 1>>
        GT_FUNCTION void increment_k(Offset const &offset = {}) {
            increment<dim::k>(offset);
        }

        /**@brief method for initializing the index */
        GT_FUNCTION void initialize(pos3<uint_t> begin, pos3<uint_t> block_no, pos3<int_t> pos_in_block) {
            using backend_t = typename IterateDomainArguments::backend_t;
            host_device::for_each_type<typename local_domain_t::strides_kinds_t>(
                initialize_index<backend_t, local_domain_t>(
                    m_local_domain.m_strides_map, begin, block_no, pos_in_block, m_index));
        }
    };
} // namespace gridtools

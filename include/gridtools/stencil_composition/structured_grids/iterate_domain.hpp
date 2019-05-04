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
#include "../positional.hpp"
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
        using local_domain_t = typename IterateDomainArguments::local_domain_t;
        GT_STATIC_ASSERT(is_local_domain<local_domain_t>::value, GT_INTERNAL_ERROR);

        GT_FUNCTION positional pos() const {
            return *(host_device::at_key<positional>(m_ptr) + host_device::at_key<positional>(m_index));
        }

        using ptr_t = typename local_domain_t::ptr_t;
        using index_t = typename local_domain_t::ptr_diff_t;

      protected:
        using iterate_domain_arguments_t = IterateDomainArguments;

        local_domain_t const &m_local_domain;

        ptr_t m_ptr;
        index_t m_index = {};

        template <class Dim, class Offset>
        GT_FUNCTION void increment(Offset const &offset) {
            sid::shift(m_index, sid::get_stride<Dim>(m_local_domain.m_strides), offset);
        }

        template <class Arg, class Accessor>
        GT_FUNCTION auto get_ptr(Accessor const &acc) const -> decay_t<decltype(host_device::at_key<Arg>(m_ptr))> {
            auto offset = host_device::at_key<Arg>(m_index);
            sid::multi_shift<Arg>(offset, m_local_domain.m_strides, acc);
            return host_device::at_key<Arg>(m_ptr) + offset;
        }

      public:
        static constexpr bool has_k_caches = false;

        GT_FUNCTION_DEVICE iterate_domain(local_domain_t const &local_domain_)
            : m_local_domain(local_domain_), m_ptr(local_domain_.m_ptr_holder()) {}

        GT_FUNCTION index_t const &index() const { return m_index; }

        /**@brief method for setting the index array
         * This method is responsible of assigning the index for the memory access at
         * the location (i,j,k). Such index is shared among all the fields contained in the
         * same storage class instance, and it is not shared among different storage instances.
         */
        GT_FUNCTION void set_index(index_t const &index) { m_index = index; }

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
            host_device::for_each_type<typename local_domain_t::esf_args_t>(
                initialize_index<typename IterateDomainArguments::backend_t, local_domain_t>(
                    m_local_domain.m_strides, begin, block_no, pos_in_block, m_ptr));
        }

        GT_FUNCTION int_t i() const { return pos().i; }

        GT_FUNCTION int_t j() const { return pos().j; }

        GT_FUNCTION int_t k() const { return pos().k; }
    };
} // namespace gridtools

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

#include <utility>

#include "../../../common/defs.hpp"
#include "../../../common/generic_metafunctions/for_each.hpp"
#include "../../../common/host_device.hpp"
#include "../../../common/hymap.hpp"
#include "../../../meta.hpp"
#include "../../dim.hpp"
#include "../../iterate_domain_aux.hpp"
#include "../../iterate_domain_fwd.hpp"
#include "../../local_domain.hpp"
#include "../../pos3.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/multi_shift.hpp"

namespace gridtools {
    /** class managing the memory accesses, indices increment
     */
    template <class LocalDomain>
    class iterate_domain_x86 {
        GT_STATIC_ASSERT(is_local_domain<LocalDomain>::value, GT_INTERNAL_ERROR);

        using index_t = typename LocalDomain::ptr_diff_t;

        LocalDomain const &m_local_domain;

        typename LocalDomain::ptr_t m_ptr;
        index_t m_index = {};
        int_t m_color = 0;

        template <class Dim, class Offset>
        GT_FORCE_INLINE void increment(Offset const &offset) {
            sid::shift(m_index, sid::get_stride<Dim>(m_local_domain.m_strides), offset);
        }

      public:
        GT_FORCE_INLINE iterate_domain_x86(LocalDomain const &local_domain_)
            : m_local_domain(local_domain_), m_ptr(local_domain_.m_ptr_holder()) {}

        GT_FORCE_INLINE index_t const &index() const { return m_index; }

        /**@brief method for setting the index array
         * This method is responsible of assigning the index for the memory access at
         * the location (i,j,k). Such index is shared among all the fields contained in the
         * same storage class instance, and it is not shared among different storage instances.
         */
        GT_FORCE_INLINE void set_index(index_t const &index) { m_index = index; }

        template <class Offset = integral_constant<int_t, 1>>
        GT_FORCE_INLINE void increment_i(Offset const &offset = {}) {
            increment<dim::i>(offset);
        }
        template <class Offset = integral_constant<int_t, 1>>
        GT_FORCE_INLINE void increment_j(Offset const &offset = {}) {
            increment<dim::j>(offset);
        }
        template <class Offset = integral_constant<int_t, 1>>
        GT_FORCE_INLINE void increment_k(Offset const &offset = {}) {
            increment<dim::k>(offset);
        }
        template <class Offset = integral_constant<int_t, 1>>
        GT_FORCE_INLINE void increment_c(Offset offset = {}) {
            m_color += offset;
            increment<dim::c>(offset);
        }

        GT_FORCE_INLINE void initialize(pos3<int_t> begin, pos3<int_t> block_no, pos3<int_t> pos_in_block) {
            for_each_type<get_keys<typename LocalDomain::ptr_t>>(
                initialize_index<typename LocalDomain::max_extent_for_tmp_t>(
                    m_local_domain.m_strides, begin, block_no, pos_in_block, m_ptr));
        }

        template <class Arg, class Accessor>
        GT_FORCE_INLINE decltype(auto) deref(Accessor const &acc) const {
            auto offset = at_key<Arg>(m_index);
            sid::multi_shift<Arg>(offset, m_local_domain.m_strides, acc);
            return *(at_key<Arg>(m_ptr) + offset);
        }
    };

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_x86<IterateDomainArguments>> : std::true_type {};
} // namespace gridtools

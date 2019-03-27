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

#include <type_traits>

#include "../../common/defs.hpp"
#include "../block.hpp"
#include "../iterate_domain_fwd.hpp"
#include "../pos3.hpp"

namespace gridtools {
    template <class Base>
    struct positional_iterate_domain : Base {
        GT_STATIC_ASSERT(is_iterate_domain<Base>::value, GT_INTERNAL_ERROR);

        struct array_index_t {
            typename Base::array_index_t index;
            pos3<int_t> pos;

            auto operator[](int_t i) const GT_AUTO_RETURN(index[i]);
            auto operator[](int_t i) GT_AUTO_RETURN(index[i]);
        };

        using Base::Base;

        template <class Offset = integral_constant<int_t, 1>>
        GT_FUNCTION void increment_i(Offset offset = {}) {
            m_pos.i += offset;
            Base::template increment_i(offset);
        }
        template <class Offset = integral_constant<int_t, 1>>
        GT_FUNCTION void increment_j(Offset offset = {}) {
            m_pos.j += offset;
            Base::template increment_j(offset);
        }
        template <class Offset = integral_constant<int_t, 1>>
        GT_FUNCTION void increment_k(Offset offset = {}) {
            m_pos.k += offset;
            Base::template increment_k(offset);
        }

        GT_FUNCTION void initialize(pos3<uint_t> begin, pos3<uint_t> block_no, pos3<int_t> pos_in_block) {
            static constexpr auto backend = typename Base::iterate_domain_arguments_t::target_t{};
            static constexpr auto block_size =
                make_pos3(block_i_size(backend), block_j_size(backend), block_k_size(backend));
            m_pos.i = begin.i + block_no.i * block_size.i + pos_in_block.i;
            m_pos.j = begin.j + block_no.j * block_size.j + pos_in_block.j;
            m_pos.k = begin.k + block_no.k * block_size.k + pos_in_block.k;
            Base::initialize(begin, block_no, pos_in_block);
        }

        GT_FUNCTION array_index_t index() const { return {Base::index(), m_pos}; }

        GT_FUNCTION void set_index(array_index_t const &index) {
            Base::set_index(index.index);
            m_pos = index.pos;
        }

        GT_FUNCTION int_t i() const { return m_pos.i; }

        GT_FUNCTION int_t j() const { return m_pos.j; }

        GT_FUNCTION int_t k() const { return m_pos.k; }

      private:
        pos3<int_t> m_pos;
    };

    template <class Base>
    struct is_iterate_domain<positional_iterate_domain<Base>> : is_iterate_domain<Base> {};

} // namespace gridtools

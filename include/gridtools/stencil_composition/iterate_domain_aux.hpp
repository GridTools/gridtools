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

#include "../common/array.hpp"
#include "../common/generic_metafunctions/for_each.hpp"
#include "../common/hymap.hpp"
#include "../meta/macros.hpp"
#include "../meta/make_indices.hpp"
#include "../meta/st_contains.hpp"
#include "../meta/st_position.hpp"
#include "../meta/type_traits.hpp"
#include "arg.hpp"
#include "block.hpp"
#include "dim.hpp"
#include "expressions/expressions.hpp"
#include "pos3.hpp"
#include "sid/concept.hpp"
#include "tmp_storage.hpp"

/**
   @file
   @brief file implementing helper functions which are used in iterate_domain to assign/increment strides, access
   indices and storage pointers.

   All the helper functions use template recursion to implement loop unrolling
*/

namespace gridtools {
    template <class StrideMaps, class LocalDomain, class PtrMap, class Backend>
    struct initialize_index_f {
        StrideMaps const &m_stride_maps;
        pos3<int_t> const &m_begin;
        pos3<int_t> const &m_block_no;
        pos3<int_t> const &m_pos_in_block;
        PtrMap &m_ptr_map;

        template <class Arg, enable_if_t<is_tmp_arg<Arg>::value, int> = 0>
        GT_FUNCTION void operator()() const {
            using storage_info_t = typename Arg::data_store_t::storage_info_t;
            GT_STATIC_ASSERT(is_storage_info<storage_info_t>::value, GT_INTERNAL_ERROR);

            host_device::at_key<Arg>(m_ptr_map) +=
                get_tmp_storage_offset<storage_info_t, typename LocalDomain::max_extent_for_tmp_t>(Backend{},
                    make_pos3<int_t>(sid::get_stride<Arg, dim::i>(m_stride_maps),
                        sid::get_stride<Arg, dim::j>(m_stride_maps),
                        sid::get_stride<Arg, dim::k>(m_stride_maps)),
                    m_block_no,
                    m_pos_in_block);
        }

        template <class Arg, enable_if_t<!is_tmp_arg<Arg>::value, int> = 0>
        GT_FUNCTION void operator()() const {
            static constexpr auto be = Backend{};
            static constexpr auto block_size = make_pos3(block_i_size(be), block_j_size(be), block_k_size(be));

            auto &ptr = host_device::at_key<Arg>(m_ptr_map);

            sid::shift(ptr,
                sid::get_stride<Arg, dim::i>(m_stride_maps),
                m_begin.i + m_block_no.i * block_size.i + m_pos_in_block.i);
            sid::shift(ptr,
                sid::get_stride<Arg, dim::j>(m_stride_maps),
                m_begin.j + m_block_no.j * block_size.j + m_pos_in_block.j);
            sid::shift(ptr,
                sid::get_stride<Arg, dim::k>(m_stride_maps),
                m_begin.k + m_block_no.k * block_size.k + m_pos_in_block.k);
        }
    };

    template <class Backend, class LocalDomain, class StrideMaps, class PtrMap>
    GT_FUNCTION initialize_index_f<StrideMaps, LocalDomain, PtrMap, Backend> initialize_index(
        StrideMaps const &stride_maps,
        pos3<int_t> const &begin,
        pos3<int_t> const &block_no,
        pos3<int_t> const &pos_in_block,
        PtrMap &ptr_map) {
        return {stride_maps, begin, block_no, pos_in_block, ptr_map};
    }
} // namespace gridtools

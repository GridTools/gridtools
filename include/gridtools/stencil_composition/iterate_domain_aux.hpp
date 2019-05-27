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
    namespace _impl {
        template <class StridesKind, class LocalDomain>
        struct get_index : meta::st_position<typename LocalDomain::strides_kinds_t, StridesKind> {};

    } // namespace _impl

    template <class LocalDomain, class Dim, class StridesMap, class ArrayIndex, class Offset>
    struct increment_index_functor {
        GT_STATIC_ASSERT((is_array_of<ArrayIndex, int>::value), GT_INTERNAL_ERROR);

        Offset const &GT_RESTRICT m_offset;
        ArrayIndex &GT_RESTRICT m_index_array;
        StridesMap const &m_strides_map;

        template <typename StridesKind>
        GT_FUNCTION void operator()() const {
            static constexpr auto index = _impl::get_index<StridesKind, LocalDomain>::value;
            GT_STATIC_ASSERT(index < ArrayIndex::size(), "Accessing an index out of bound in fusion tuple");
            sid::shift(
                m_index_array[index], sid::get_stride<Dim>(host_device::at_key<StridesKind>(m_strides_map)), m_offset);
        }
    };

    template <class Dim, class LocalDomain, class StridesMap, class ArrayIndex, class Offset>
    GT_FUNCTION void do_increment(
        Offset const &GT_RESTRICT offset, StridesMap const &strides_map, ArrayIndex &GT_RESTRICT index) {
        host_device::for_each_type<typename LocalDomain::strides_kinds_t>(
            increment_index_functor<LocalDomain, Dim, StridesMap, ArrayIndex, Offset>{offset, index, strides_map});
    }

    /**@brief functor initializing the indices does the actual assignment
     *     This method is responsible of computing the index for the memory access at
     *     the location (i,j,k). Such index is shared among all the fields contained in the
     *     same storage class instance, and it is not shared among different storage instances.
     * @tparam Coordinate direction along which the increment takes place
     * @tparam StridesCached strides cached type
     * @tparam StorageSequence sequence of storages
     */
    template <class StridesKind, class MaxExtent, bool IsTmp>
    struct get_index_offset_f;

    template <class StridesKind, class MaxExtent>
    struct get_index_offset_f<StridesKind, MaxExtent, false> {
        template <class Backend, class Stride, class Begin, class BlockNo, class PosInBlock>
        GT_FUNCTION int_t operator()(Backend const &,
            Stride const &GT_RESTRICT stride,
            Begin const &GT_RESTRICT begin,
            BlockNo const &GT_RESTRICT block_no,
            PosInBlock const &GT_RESTRICT pos_in_block) const {
            static constexpr auto block_size =
                make_pos3(block_i_size(Backend{}), block_j_size(Backend{}), block_k_size(Backend{}));
            return stride.i * (begin.i + block_no.i * block_size.i + pos_in_block.i) +
                   stride.j * (begin.j + block_no.j * block_size.j + pos_in_block.j) +
                   stride.k * (begin.k + block_no.k * block_size.k + pos_in_block.k);
        }
    };

    template <class StridesKind, class MaxExtent>
    struct get_index_offset_f<StridesKind, MaxExtent, true> {
        template <class Backend, class Stride, class Begin, class BlockNo, class PosInBlock>
        GT_FUNCTION int_t operator()(Backend const &backend,
            Stride const &GT_RESTRICT stride,
            Begin const &GT_RESTRICT /*begin*/,
            BlockNo const &GT_RESTRICT block_no,
            PosInBlock const &GT_RESTRICT pos_in_block) const {
            return get_tmp_storage_offset<StridesKind, MaxExtent>(backend, stride, block_no, pos_in_block);
        }
    };

    template <class StridesMap, class LocalDomain, class ArrayIndex, class Backend>
    struct initialize_index_f {
        GT_STATIC_ASSERT((is_array_of<ArrayIndex, int>::value), GT_INTERNAL_ERROR);
        StridesMap const &m_strides_map;
        pos3<int_t> const &m_begin;
        pos3<int_t> const &m_block_no;
        pos3<int_t> const &m_pos_in_block;
        ArrayIndex &m_index_array;

        template <typename StridesKind>
        GT_FUNCTION void operator()() const {
            static constexpr auto index = _impl::get_index<StridesKind, LocalDomain>::value;
            GT_STATIC_ASSERT(index < ArrayIndex::size(), "Accessing an index out of bound in fusion tuple");
            static constexpr auto backend = Backend{};
            static constexpr auto is_tmp =
                meta::st_contains<typename LocalDomain::tmp_strides_kinds_t, StridesKind>::value;
            auto const &strides = host_device::at_key<StridesKind>(m_strides_map);
            m_index_array[index] =
                get_index_offset_f<StridesKind, typename LocalDomain::max_extent_for_tmp_t, is_tmp>{}(backend,
                    make_pos3<int>(sid::get_stride<dim::i>(strides),
                        sid::get_stride<dim::j>(strides),
                        sid::get_stride<dim::k>(strides)),
                    m_begin,
                    m_block_no,
                    m_pos_in_block);
        }
    };

    template <class Backend, class LocalDomain, class StridesMap, class ArrayIndex>
    GT_FUNCTION initialize_index_f<StridesMap, LocalDomain, ArrayIndex, Backend> initialize_index(
        StridesMap const &strides_map,
        pos3<int_t> const &begin,
        pos3<int_t> const &block_no,
        pos3<int_t> const &pos_in_block,
        ArrayIndex &index_array) {
        return {strides_map, begin, block_no, pos_in_block, index_array};
    }
} // namespace gridtools

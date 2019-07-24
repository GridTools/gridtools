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

#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/tuple_util.hpp"
#include "../arg.hpp"
#include "../dim.hpp"
#include "../extent.hpp"
#include "../sid/blocked_dim.hpp"
#include "../sid/contiguous.hpp"
#include "../sid/sid_shift_origin.hpp"

namespace gridtools {
    namespace cuda {
        namespace tmp_impl_ {
            template <class Extent>
            hymap::keys<dim::i, dim::j>::values<integral_constant<int_t, -Extent::iminus::value>,
                integral_constant<int_t, -Extent::jminus::value>>
            origin_offset(Extent) {
                return {};
            }

            template <class Plh,
                class IBlockSize,
                class JBlockSize,
                class Extent,
                std::enable_if_t<Plh::location_t::n_colors::value == 1, int> = 0>
            auto sizes(Plh,
                IBlockSize i_block_size,
                JBlockSize j_block_size,
                Extent,
                int_t n_blocks_i,
                int_t n_blocks_j,
                int_t k_size) {
                return tuple_util::make<
                    hymap::keys<dim::i, dim::j, sid::blocked_dim<dim::i>, sid::blocked_dim<dim::j>, dim::k>::values>(
                    i_block_size - typename Extent::iminus() + typename Extent::iplus(),
                    j_block_size - typename Extent::jminus() + typename Extent::jplus(),
                    n_blocks_i,
                    n_blocks_j,
                    k_size);
            }

            template <class Plh,
                class IBlockSize,
                class JBlockSize,
                class Extent,
                std::enable_if_t<(Plh::location_t::n_colors::value > 1), int> = 0>
            auto sizes(Plh,
                IBlockSize i_block_size,
                JBlockSize j_block_size,
                Extent,
                int_t n_blocks_i,
                int_t n_blocks_j,
                int_t k_size) {
                return tuple_util::make<
                    hymap::keys<dim::i, dim::j, dim::c, sid::blocked_dim<dim::i>, sid::blocked_dim<dim::j>, dim::k>::
                        values>(i_block_size - typename Extent::iminus() + typename Extent::iplus(),
                    j_block_size - typename Extent::jminus() + typename Extent::jplus(),
                    integral_constant<int_t, Plh::location_t::n_colors::value>{},
                    n_blocks_i,
                    n_blocks_j,
                    k_size);
            }
        } // namespace tmp_impl_

        template <class Plh, class BlockSizeI, class BlockSizeJ, class Extent, class Allocator>
        auto make_tmp_storage(Plh plh,
            BlockSizeI block_size_i,
            BlockSizeJ block_size_j,
            Extent extent,
            int_t n_blocks_i,
            int_t n_blocks_j,
            int_t k_size,
            Allocator &alloc) {
            GT_STATIC_ASSERT(is_plh<Plh>::value, GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
            return sid::shift_sid_origin(
                sid::make_contiguous<typename Plh::data_t, int_t>(
                    alloc, tmp_impl_::sizes(plh, block_size_i, block_size_j, extent, n_blocks_i, n_blocks_j, k_size)),
                tmp_impl_::origin_offset(extent));
        }
    } // namespace cuda
} // namespace gridtools

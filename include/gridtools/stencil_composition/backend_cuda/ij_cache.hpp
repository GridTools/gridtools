/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../../common/defs.hpp"
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../../meta.hpp"
#include "../arg.hpp"
#include "../dim.hpp"
#include "../extent.hpp"
#include "../sid/contiguous.hpp"
#include "../sid/sid_shift_origin.hpp"
#include "shared_allocator.hpp"

namespace gridtools {
    namespace cuda {
        namespace ij_cache_impl_ {
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
            auto sizes(Plh, IBlockSize i_block_size, JBlockSize j_block_size, Extent) {
                return tuple_util::make<hymap::keys<dim::i, dim::j>::values>(
                    i_block_size - typename Extent::iminus{} + typename Extent::iplus{},
                    j_block_size - typename Extent::jminus{} + typename Extent::jplus{});
            }

            template <class Plh,
                class IBlockSize,
                class JBlockSize,
                class Extent,
                std::enable_if_t<(Plh::location_t::n_colors::value > 1), int> = 0>
            auto sizes(Plh, IBlockSize i_block_size, JBlockSize j_block_size, Extent) {
                return tuple_util::make<hymap::keys<dim::i, dim::c, dim::j>::values>(
                    i_block_size - typename Extent::iminus{} + typename Extent::iplus{},
                    integral_constant<int_t, Plh::location_t::n_colors::value>{},
                    j_block_size - typename Extent::jminus{} + typename Extent::jplus{});
            }

            template <class Plh, class BlockSizeI, class BlockSizeJ, class Extent>
            auto make_ij_cache(
                Plh plh, BlockSizeI block_size_i, BlockSizeJ block_size_j, Extent extent, shared_allocator &alloc) {
                GT_STATIC_ASSERT(is_plh<Plh>::value, GT_INTERNAL_ERROR);
                GT_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
                return sid::shift_sid_origin(sid::make_contiguous<typename Plh::data_store_t::data_t, int_t>(
                                                 alloc, sizes(plh, block_size_i, block_size_j, extent)),
                    origin_offset(extent));
            }
        } // namespace ij_cache_impl_

        using ij_cache_impl_::make_ij_cache;
    } // namespace cuda
} // namespace gridtools

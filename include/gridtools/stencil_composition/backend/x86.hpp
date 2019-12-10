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

#include <memory>
#include <utility>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/host_device.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/omp.hpp"
#include "../../common/tuple.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../dim.hpp"
#include "../sid/allocator.hpp"
#include "../sid/as_const.hpp"
#include "../sid/block.hpp"
#include "../sid/composite.hpp"
#include "../sid/concept.hpp"
#include "../sid/contiguous.hpp"
#include "../sid/loop.hpp"
#include "../sid/sid_shift_origin.hpp"
#include "../stage_matrix.hpp"

namespace gridtools {
    namespace x86 {
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1900
        template <class DataStores>
        struct make_sid_f {
            DataStores &m_data_stores;
            template <class PtrInfo>
            auto operator()(PtrInfo) const {
                return sid::add_const(PtrInfo::is_const(), at_key<decltype(PtrInfo::plh())>(m_data_stores));
            }
        };

        template <class Stage, class Sizes, class ShiftBack>
        struct k_loop_f {
            Sizes m_sizes;
            ShiftBack m_shift_back;

            template <class Ptr, class Strides>
            GT_FORCE_INLINE void operator()(Ptr &ptr, Strides const &strides) const {
                tuple_util::for_each(
                    [&ptr, &strides](auto cell, auto size) {
                        for (int_t k = 0; k < size; ++k) {
                            cell(ptr, strides);
                            cell.inc_k(ptr, strides);
                        }
                    },
                    Stage::cells(),
                    m_sizes);
                sid::shift(ptr, sid::get_stride<dim::k>(strides), m_shift_back);
            }
        };
#endif

        template <class Stage, class Grid, class DataStores>
        auto make_stage_loop(Stage, Grid const &grid, DataStores &data_stores) {
            using extent_t = typename Stage::extent_t;

            using plh_map_t = typename Stage::plh_map_t;
            using keys_t = meta::rename<sid::composite::keys, meta::transform<meta::first, plh_map_t>>;
            auto composite = tuple_util::convert_to<keys_t::template values>(tuple_util::transform(
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1900
                make_sid_f<decltype(data_stores)> { data_stores }
#else
                [&](auto info) { return sid::add_const(info.is_const(), at_key<decltype(info.plh())>(data_stores)); }
#endif
                ,
                Stage::plh_map()));
            using ptr_diff_t = sid::ptr_diff_type<decltype(composite)>;

            auto strides = sid::get_strides(composite);
            ptr_diff_t offset{};
            sid::shift(offset, sid::get_stride<dim::i>(strides), extent_t::minus(dim::i()));
            sid::shift(offset, sid::get_stride<dim::j>(strides), extent_t::minus(dim::j()));
            sid::shift(offset, sid::get_stride<dim::k>(strides), grid.k_start(Stage::interval(), Stage::execution()));

            auto shift_back = -grid.k_size(Stage::interval()) * Stage::k_step();
            auto k_sizes =
                tuple_util::transform([&](auto cell) { return grid.k_size(cell.interval()); }, Stage::cells());
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1900
            k_loop_f<Stage, decltype(k_sizes), decltype(shift_back)> k_loop{std::move(k_sizes), shift_back};
#else
            auto k_loop = [k_sizes = std::move(k_sizes), shift_back](auto &ptr, auto const &strides) {
                tuple_util::for_each(
                    [&ptr, &strides](auto cell, auto size) {
                        for (int_t k = 0; k < size; ++k) {
                            cell(ptr, strides);
                            cell.inc_k(ptr, strides);
                        }
                    },
                    Stage::cells(),
                    k_sizes);
                sid::shift(ptr, sid::get_stride<dim::k>(strides), shift_back);
            };
#endif
            return [origin = sid::get_origin(composite) + offset,
                       strides = std::move(strides),
                       k_loop = std::move(k_loop)](int_t i_block, int_t j_block, int_t i_size, int_t j_size) {
                ptr_diff_t offset{};
                sid::shift(offset, sid::get_stride<dim::thread>(strides), omp_get_thread_num());
                sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::i>>(strides), i_block);
                sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::j>>(strides), j_block);
                auto ptr = origin() + offset;
                auto i_loop = sid::make_loop<dim::i>(extent_t::extend(dim::i(), i_size));
                auto j_loop = sid::make_loop<dim::j>(extent_t::extend(dim::j(), j_size));
                i_loop(j_loop(k_loop))(ptr, strides);
            };
        }

        template <class IBlockSize = integral_constant<int_t, 8>, class JBlockSize = integral_constant<int_t, 8>>
        struct backend {};

        template <class IBlockSize, class JBlockSize, class Spec, class Grid, class DataStores>
        void gridtools_backend_entry_point(
            backend<IBlockSize, JBlockSize>, Spec, Grid const &grid, DataStores external_data_stores) {
            using stages_t = stage_matrix::make_split_view<Spec>;

            auto alloc = sid::make_cached_allocator(&std::make_unique<char[]>);

            using tmp_plh_map_t = stage_matrix::remove_caches_from_plh_map<typename stages_t::tmp_plh_map_t>;
            auto temporaries = stage_matrix::make_data_stores(tmp_plh_map_t(), [&grid, &alloc](auto info) {
                auto extent = info.extent();
                auto interval = stages_t::interval();
                auto num_colors = info.num_colors();
                auto offsets = tuple_util::make<hymap::keys<dim::i, dim::j, dim::k>::values>(
                    -extent.minus(dim::i()), -extent.minus(dim::j()), -grid.k_start(interval) - extent.minus(dim::k()));
                auto sizes =
                    tuple_util::make<hymap::keys<dim::c, dim::k, dim::j, dim::i, dim::thread>::values>(num_colors,
                        grid.k_size(interval, extent),
                        extent.extend(dim::j(), JBlockSize()),
                        extent.extend(dim::i(), IBlockSize()),
                        omp_get_max_threads());

                using stride_kind = meta::list<decltype(extent), decltype(num_colors)>;
                return sid::shift_sid_origin(
                    sid::make_contiguous<decltype(info.data()), int_t, stride_kind>(alloc, sizes), offsets);
            });

            auto blocked_external_data_stores = tuple_util::transform(
                [&](auto &&data_store) {
                    return sid::block(std::forward<decltype(data_store)>(data_store),
                        hymap::keys<dim::i, dim::j>::values<IBlockSize, JBlockSize>());
                },
                std::move(external_data_stores));

            auto data_stores = hymap::concat(std::move(blocked_external_data_stores), std::move(temporaries));

            auto stage_loops = tuple_util::transform(
                [&](auto stage) { return make_stage_loop(stage, grid, data_stores); }, meta::rename<tuple, stages_t>());

            int_t total_i = grid.i_size();
            int_t total_j = grid.j_size();

            int_t NBI = (total_i + IBlockSize::value - 1) / IBlockSize::value;
            int_t NBJ = (total_j + JBlockSize::value - 1) / JBlockSize::value;

#pragma omp parallel for collapse(2)
            for (int_t bi = 0; bi < NBI; ++bi) {
                for (int_t bj = 0; bj < NBJ; ++bj) {
                    int_t i_size = bi + 1 == NBI ? total_i - bi * IBlockSize::value : IBlockSize::value;
                    int_t j_size = bj + 1 == NBJ ? total_j - bj * JBlockSize::value : JBlockSize::value;
                    tuple_util::for_each([=](auto &&fun) { fun(bi, bj, i_size, j_size); }, stage_loops);
                }
            }
        }
    } // namespace x86
} // namespace gridtools

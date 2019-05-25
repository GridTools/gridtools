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

#include <cstdlib>
#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/hymap.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../../storage/common/storage_info.hpp"
#include "../arg.hpp"
#include "../dim.hpp"
#include "../grid.hpp"
#include "../local_domain.hpp"
#include "../mss_components.hpp"
#include "../sid/concept.hpp"
#include "../sid/loop.hpp"

namespace gridtools {
    namespace naive_impl_ {
        template <class Strides, class Grid>
        struct correct_ptr_f {
            Strides const &m_strides;
            Grid const &m_grid;

            template <class Arg, class Dim>
            auto stride() const {
                return sid::get_stride<Arg, Dim>(m_strides);
            }

            template <class Arg, class Ptr, std::enable_if_t<is_tmp_arg<Arg>::value, int> = 0>
            void operator()(Ptr &ptr) const {
                using storage_info_t = typename Arg::data_store_t::storage_info_t;
                GT_STATIC_ASSERT(is_storage_info<storage_info_t>::value, GT_INTERNAL_ERROR);
                sid::shift(ptr, stride<Arg, dim::i>(), storage_info_t::halo_t::template at<dim::i::value>());
                sid::shift(ptr, stride<Arg, dim::j>(), storage_info_t::halo_t::template at<dim::j::value>());
                sid::shift(ptr, stride<Arg, dim::k>(), -m_grid.k_min());
            }

            template <class Arg, class Ptr, std::enable_if_t<!is_tmp_arg<Arg>::value, int> = 0>
            void operator()(Ptr &ptr) const {
                sid::shift(ptr, stride<Arg, dim::i>(), m_grid.i_low_bound());
                sid::shift(ptr, stride<Arg, dim::j>(), m_grid.j_low_bound());
            }
        };
        template <class Ptr, class Strides, class Grid>
        void correct_ptr(Ptr &ptr, Strides const &strides, Grid const &grid) {
            hymap::for_each(correct_ptr_f<Strides, Grid>{strides, grid}, ptr);
        }
    } // namespace naive_impl_

    /**
     * @brief loops over all blocks and execute sequentially all mss functors
     * @tparam MssComponents a meta array with the mss components of all MSS
     */
    template <class MssComponents, class LocalDomains, class Grid>
    void fused_mss_loop(backend::naive, LocalDomains const &local_domains, Grid const &grid) {
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        tuple_util::for_each(
            [&grid](auto mss_components, auto const &local_domain) {
                using mss_components_t = decltype(mss_components);
                GT_STATIC_ASSERT(is_local_domain<std::decay_t<decltype(local_domain)>>::value, GT_INTERNAL_ERROR);
                GT_STATIC_ASSERT(is_mss_components<mss_components_t>::value, GT_INTERNAL_ERROR);
                for_each<typename mss_components_t::loop_intervals_t>([&grid, &local_domain](auto loop_interval) {
                    using loop_interval_t = decltype(loop_interval);
                    using from_t = meta::first<loop_interval_t>;
                    using to_t = meta::second<loop_interval_t>;
                    using stages_t = meta::flatten<meta::third<loop_interval_t>>;
                    for_each<stages_t>([&grid, &local_domain](auto stage) {
                        auto ptr = local_domain.m_ptr_holder();
                        auto const &strides = local_domain.m_strides;
                        naive_impl_::correct_ptr(ptr, strides, grid);

                        using extent_t = typename decltype(stage)::extent_t;

                        sid::shift(ptr, sid::get_stride<dim::i>(strides), typename extent_t::iminus{});
                        sid::shift(ptr, sid::get_stride<dim::j>(strides), typename extent_t::jminus{});
                        sid::shift(ptr, sid::get_stride<dim::k>(strides), grid.template value_at<from_t>());

                        auto i_loop = sid::make_loop<dim::i>(grid.i_high_bound() - grid.i_low_bound() + 1 +
                                                             extent_t::iplus::value - extent_t::iminus::value);
                        auto j_loop = sid::make_loop<dim::j>(grid.j_high_bound() - grid.j_low_bound() + 1 +
                                                             extent_t::jplus::value - extent_t::jminus::value);
                        auto k_loop = sid::make_loop<dim::k>(grid.count(from_t{}, to_t{}),
                            execute::step<typename mss_components_t::mss_descriptor_t::execution_engine_t>);

                        i_loop(j_loop(k_loop(stage)))(ptr, strides);
                    });
                });
            },
            MssComponents{},
            local_domains);
    }

    /**
     * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
     */
    std::true_type mss_fuse_esfs(backend::naive);
} // namespace gridtools

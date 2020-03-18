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
#include <utility>

#include "../../../common/cuda_type_traits.hpp"
#include "../../../common/cuda_util.hpp"
#include "../../../common/defs.hpp"
#include "../../../common/hymap.hpp"
#include "../../../common/integral_constant.hpp"
#include "../../../common/tuple_util.hpp"
#include "../../../meta.hpp"
#include "../../../sid/allocator.hpp"
#include "../../../sid/as_const.hpp"
#include "../../../sid/block.hpp"
#include "../../../sid/blocked_dim.hpp"
#include "../../../sid/composite.hpp"
#include "../../../sid/concept.hpp"
#include "../../../sid/contiguous.hpp"
#include "../../../sid/sid_shift_origin.hpp"
#include "../../be_api.hpp"
#include "../../common/caches.hpp"
#include "../../common/dim.hpp"
#include "../../common/extent.hpp"
#include "j_cache.hpp"
#include "launch_kernel.hpp"
#include "make_kernel_fun.hpp"

namespace gridtools {
    namespace cuda2 {
        template <class Keys>
        struct deref_f {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
            template <class Key, class T>
            GT_FUNCTION std::enable_if_t<is_texture_type<T>::value && meta::st_contains<Keys, Key>::value, T>
            operator()(Key, T const *ptr) const {
                return __ldg(ptr);
            }
#endif
            template <class Key, class Ptr>
            GT_FUNCTION decltype(auto) operator()(Key, Ptr ptr) const {
                return *ptr;
            }
        };

        template <class IBlockSize = integral_constant<int_t, 64>,
            class JBlockSize = integral_constant<int_t, 8>,
            class KBlockSize = integral_constant<int_t, 1>>
        struct backend {
            template <class DataStoreMap>
            static auto block(DataStoreMap data_stores) {
                return tuple_util::transform(
                    [=](auto &&src) {
                        return sid::block(std::forward<decltype(src)>(src),
                            tuple_util::make<hymap::keys<dim::i, dim::j>::values>(IBlockSize(), JBlockSize()));
                    },
                    std::move(data_stores));
            }

            template <class DataStores>
            struct make_sid_f {
                DataStores &m_data_stores;

                template <class PlhInfo, std::enable_if_t<PlhInfo::is_tmp_t::value, int> = 0>
                auto operator()(PlhInfo) const {
                    return j_cache_sid_t<typename PlhInfo::extent_t>();
                }

                template <class PlhInfo, std::enable_if_t<!PlhInfo::is_tmp_t::value, int> = 0>
                auto operator()(PlhInfo) const {
                    return sid::add_const(PlhInfo::is_const(),
                        sid::block(at_key<typename PlhInfo::plh_t>(m_data_stores),
                            tuple_util::make<hymap::keys<dim::i, dim::j>::values>(IBlockSize(), JBlockSize())));
                }
            };

            template <class Spec, class Grid, class DataStores>
            static void entry_point(Grid const &grid, DataStores data_stores) {
                using msses_t = be_api::make_fused_view<Spec>;
                static_assert(meta::length<msses_t>::value == 1, "Not implemented");
                using mss_t = meta::first<msses_t>;
                static_assert(be_api::is_parallel<typename mss_t::execution_t>(), "Not implemented");
                using const_keys_t =
                    meta::transform<be_api::get_key, meta::filter<be_api::get_is_const, typename mss_t::plh_map_t>>;
                using deref_t = deref_f<const_keys_t>;

                using plh_map_t = typename mss_t::plh_map_t;
                using keys_t = meta::rename<sid::composite::keys, meta::transform<meta::first, plh_map_t>>;

                auto composite = tuple_util::convert_to<keys_t::template values>(
                    tuple_util::transform(make_sid_f<DataStores>{data_stores}, plh_map_t()));

                launch_kernel<typename mss_t::extent_t, IBlockSize::value, JBlockSize::value>(grid.i_size(),
                    grid.j_size(),
                    (grid.k_size() + KBlockSize::value - 1) / KBlockSize::value,
                    make_kernel_fun<deref_t, mss_t, JBlockSize::value, KBlockSize::value>(grid, composite));
            }

            template <class Spec, class Grid, class DataStores>
            friend void gridtools_backend_entry_point(backend, Spec, Grid const &grid, DataStores data_stores) {
                return backend::entry_point<Spec>(grid, std::move(data_stores));
            }
        };
    } // namespace cuda2
} // namespace gridtools

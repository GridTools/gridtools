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

#include <array>
#include <ostream>

#include <boost/a>
#include <nlohmann/json.hpp>

#include "../common/defs.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "be_api.hpp"
#include "common/caches.hpp"
#include "common/extent.hpp"
#include "core/execution_types.hpp"
#include "core/interval.hpp"
#include "core/level.hpp"

namespace gridtools {
    namespace stencil {
        namespace dump_backend {

            inline auto from_execution(core::parallel) { return "parallel"; }
            inline auto from_execution(core::backward) { return "backward"; }
            inline auto from_execution(core::forward) { return "forward"; }
            inline auto from_cache(cache_type::ij) { return "ij"; }
            inline auto from_cache(cache_type::k) { return "k"; }
            inline auto from_cache_io_policy(cache_io_policy::fill) { return "fill"; }
            inline auto from_cache_io_policy(cache_io_policy::flush) { return "flush"; }

            template <int... Is>
            auto from_extent(extent<Is...>) {
                return tuple_util::make<std::array, int>(Is...);
            }

            template <uint_t Splitter, int_t Offset, int_t Limit>
            auto from_level(core::level<Splitter, Offset, Limit>) {
                return tuple_util::make<std::array, int>(Splitter, Offset);
            }

            template <class From, class To>
            auto from_interval(core::interval<From, To>) {
                return tuple_util::make<std::array>(from_level(From()), from_level(To()));
            }

            template <class Fun>
            nlohmann::json from_fun(Fun) {
                return {};
            }

            template <class Plh,
                class... Caches,
                class IsTmp,
                class Data,
                class NumColors,
                class IsConst,
                class Extent,
                class... CacheIoPolicies>
            nlohmann::json from_plh_info(be_api::plh_info<meta::list<Plh, Caches...>,
                IsTmp,
                Data,
                NumColors,
                IsConst,
                Extent,
                meta::list<CacheIoPolicies...>>) {
                return {{"plh", "bar"},
                    {"caches", {from_cache(Caches())...}},
                    {"is_tmp", IsTmp::value},
                    {"data_type", "foo"},
                    {"num_colors", NumColors::value},
                    {"is_const", IsConst::value},
                    {"extent", from_extent(Extent())},
                    {"cache_io_policies", {from_cache_io_policy(CacheIoPolicies())...}}};
            }

            template <template <class...> class L,
                template <class...> class LL,
                class... Funs,
                class Interval,
                class... PlhInfos,
                class Extent,
                class Execution,
                class NeedSync>
            nlohmann::json from_cell(be_api::cell<L<Funs...>, Interval, LL<PlhInfos...>, Extent, Execution, NeedSync>) {
                return {{"funs", {from_fun(Funs())...}},
                    {"interval", from_interval(Interval())},
                    {"plh_infos", {from_plh_info(PlhInfos())...}},
                    {"extent", from_extent(Extent())},
                    {"execution", from_execution(Execution())},
                    {"need_sync", NeedSync::value}};
            }

            template <template <class...> class L, class... Cells>
            nlohmann::json from_row(L<Cells...>) {
                return {{"cells", {from_cell(Cells())...}}};
            }

            template <template <class...> class L, class... Rows>
            nlohmann::json from_matrix(L<Rows...>) {
                return {{"rows", {from_row(Rows())...}}};
            }

            template <template <class...> class L, class... Matrices>
            nlohmann::json from_matrices(L<Matrices...>) {
                return {{"matrices", {from_matrix(Matrices())...}}};
            }

            struct dump {
                std::ostream &m_sink;
                template <class Spec, class... Ts>
                friend void gridtools_backend_entry_point(dump obj, Spec spec, Ts &&...) {
                    obj.m_sink << from_matrices(spec) << std::endl;
                }
            };
        } // namespace dump_backend
        using dump_backend::dump;
    } // namespace stencil
} // namespace gridtools

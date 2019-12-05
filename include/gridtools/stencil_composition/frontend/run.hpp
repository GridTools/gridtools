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

#include <boost/preprocessor/punctuation/remove_parens.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

#include "../../common/hymap.hpp"
#include "../../meta.hpp"
#include "../arg.hpp"
#include "../backend.hpp"
#include "../caches/cache_traits.hpp"
#include "../compute_extents_metafunctions.hpp"
#include "../esf.hpp"
#include "../execution_types.hpp"
#include "../mss.hpp"

namespace gridtools {
    namespace frontend_impl_ {
        template <class...>
        struct spec {};

        template <class ExecutionType, class Esfs, class Caches>
        struct spec<mss_descriptor<ExecutionType, Esfs, Caches>> {
            template <class F, class... Args>
            constexpr spec<mss_descriptor<ExecutionType,
                meta::push_back<Esfs, esf_descriptor<F, meta::list<Args...>, void>>,
                Caches>>
            stage(F, Args...) const {
                return {};
            }

            template <class Extent, class F, class... Args>
            constexpr spec<mss_descriptor<ExecutionType,
                meta::push_back<Esfs, esf_descriptor<F, meta::list<Args...>, Extent>>,
                Caches>>
            stage_with_extent(Extent, F, Args...) const {
                return {};
            }
        };

        template <class ExecutionType, class... Caches>
        struct empty_spec : spec<mss_descriptor<ExecutionType, meta::list<>, cache_map<Caches...>>> {
            template <class... CacheIOPolicies, class CacheType, class... Args>
            constexpr empty_spec<ExecutionType,
                Caches...,
                cache_info<Args, meta::list<CacheType>, meta::list<CacheIOPolicies...>>...>
            cached(CacheType, Args...) const {
                return {};
            }
            template <class... Args>
            constexpr empty_spec<ExecutionType,
                Caches...,
                cache_info<Args, meta::list<cache_type::ij>, meta::list<>>...>
            ij_cached(Args...) const {
                return {};
            }
            template <class... Args>
            constexpr empty_spec<ExecutionType, Caches..., cache_info<Args, meta::list<cache_type::k>, meta::list<>>...>
            k_cached(Args...) const {
                return {};
            }
            template <class... Args>
            constexpr empty_spec<ExecutionType,
                Caches...,
                cache_info<Args, meta::list<cache_type::k>, meta::list<cache_io_policy::flush>>...>
            k_cached(cache_io_policy::flush, Args...) const {
                return {};
            }
            template <class... Args>
            constexpr empty_spec<ExecutionType,
                Caches...,
                cache_info<Args, meta::list<cache_type::k>, meta::list<cache_io_policy::fill>>...>
            k_cached(cache_io_policy::fill, Args...) const {
                return {};
            }
            template <class... Args>
            constexpr empty_spec<ExecutionType,
                Caches...,
                cache_info<Args,
                    meta::list<cache_type::k>,
                    meta::list<cache_io_policy::fill, cache_io_policy::flush>>...>
            k_cached(cache_io_policy::fill, cache_io_policy::flush, Args...) const {
                return {};
            }
            template <class... Args>
            constexpr empty_spec<ExecutionType,
                Caches...,
                cache_info<Args,
                    meta::list<cache_type::k>,
                    meta::list<cache_io_policy::fill, cache_io_policy::flush>>...>
            k_cached(cache_io_policy::flush, cache_io_policy::fill, Args...) const {
                return {};
            }
        };

        template <int_t Size = GT_DEFAULT_VERTICAL_BLOCK_SIZE>
        constexpr empty_spec<execute::parallel_block<Size>> execute_parallel() {
            return {};
        }
        constexpr empty_spec<execute::forward> execute_forward() { return {}; }
        constexpr empty_spec<execute::backward> execute_backward() { return {}; }

        template <class... Msses>
        constexpr spec<Msses...> multi_pass(spec<Msses>...) {
            return {};
        }

        template <size_t>
        struct arg {};

        template <class Comp, class Backend, class Grid, class... Fields, size_t... Is>
        void run_impl(Comp comp, Backend, Grid const &grid, std::index_sequence<Is...>, Fields &&... fields) {
            using spec_t = decltype(comp(arg<Is>()...));
            using entry_point_t = backend_entry_point_f<Backend, spec_t>;
            using data_store_map_t = typename hymap::keys<arg<Is>...>::template values<Fields &...>;
            entry_point_t()(grid, data_store_map_t{fields...});
        }

        template <class Comp, class Backend, class Grid, class... Fields>
        void run(Comp comp, Backend be, Grid const &grid, Fields &&... fields) {
            run_impl(comp, be, grid, std::index_sequence_for<Fields...>(), std::forward<Fields>(fields)...);
        }

        template <class F, class Backend, class Grid, class... Fields>
        void easy_run(F, Backend be, Grid const &grid, Fields &&... fields) {
            return run([](auto... args) { return execute_parallel().stage(F(), args...); },
                be,
                grid,
                std::forward<Fields>(fields)...);
        }

        template <class... Msses, class Arg>
        constexpr lookup_extent_map<get_extent_map_from_msses<spec<Msses...>>, Arg> get_arg_extent(
            spec<Msses...>, Arg) {
            return {};
        }

        template <class Mss>
        using rw_args_from_mss = compute_readwrite_args<typename Mss::esf_sequence_t>;

        template <class Msses,
            class RwArgsLists = meta::transform<rw_args_from_mss, Msses>,
            class RawRwArgs = meta::flatten<RwArgsLists>>
        using all_rw_args = meta::dedup<RawRwArgs>;

        template <class... Msses,
            class Arg,
            class RwPlhs = all_rw_args<spec<Msses...>>,
            intent Intent = meta::st_contains<RwPlhs, Arg>::value ? intent::inout : intent::in>
        constexpr std::integral_constant<intent, Intent> get_arg_intent(spec<Msses...>, Arg) {
            return {};
        }
    } // namespace frontend_impl_
    using frontend_impl_::easy_run;
    using frontend_impl_::execute_backward;
    using frontend_impl_::execute_forward;
    using frontend_impl_::execute_parallel;
    using frontend_impl_::get_arg_extent;
    using frontend_impl_::get_arg_intent;
    using frontend_impl_::multi_pass;
    using frontend_impl_::run;

    template <size_t, class NumColors, class Data>
    struct colored_tmp_arg {
        using data_t = Data;
        using num_colors_t = NumColors;
        using tmp_tag = std::true_type;
    };
} // namespace gridtools

#define GT_INTERNAL_DECLARE_TMP(r, type, name) \
    constexpr ::gridtools::tmp_arg<__COUNTER__, BOOST_PP_REMOVE_PARENS(type)> name = {};

#define GT_DECLARE_TMP(type, ...)                                                               \
    BOOST_PP_SEQ_FOR_EACH(GT_INTERNAL_DECLARE_TMP, type, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)) \
    static_assert(1, "")

#define GT_INTERNAL_DECLARE_COLORED_TMP(r, type_location, name)                              \
    constexpr ::gridtools::colored_tmp_arg<__COUNTER__,                                      \
        typename BOOST_PP_REMOVE_PARENS(BOOST_PP_TUPLE_ELEM(2, 1, type_location))::n_colors, \
        BOOST_PP_REMOVE_PARENS(BOOST_PP_TUPLE_ELEM(2, 0, type_location))>                    \
        name = {};

#define GT_DECLARE_COLORED_TMP(type, location, ...)                                                                 \
    BOOST_PP_SEQ_FOR_EACH(GT_INTERNAL_DECLARE_COLORED_TMP, (type, location), BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)) \
    static_assert(1, "")

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

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../caches/cache_metafunctions.hpp"
#include "../caches/cache_traits.hpp"
#include "../caches/extract_extent_caches.hpp"
#include "../dim.hpp"
#include "../global_parameter.hpp"
#include "../positional.hpp"
#include "../sid/concept.hpp"
#include "loop_interval.hpp"

namespace gridtools {
    namespace cuda {
        namespace fill_flush_impl_ {
            template <class>
            struct k_cache_original {};

            template <class Plh, class BoundChecker, class Sync, class From, class Count, class Step>
            struct sync_all_stage {
                template <class Ptrs, class Strides, class ExtentValidator>
                GT_FUNCTION_DEVICE void operator()(Ptrs const &GT_RESTRICT ptrs,
                    Strides const &GT_RESTRICT strides,
                    ExtentValidator &&extent_validator) const {
                    auto k_pos = *device::at_key<positional<dim::k>>(ptrs) + From();
                    auto &&bound_checker = *device::at_key<BoundChecker>(ptrs);

                    auto original = device::at_key<k_cache_original<Plh>>(ptrs);
                    sid::shift(original, sid::get_stride_element<k_cache_original<Plh>, dim::k>(strides), From());

                    auto cache = device::at_key<Plh>(ptrs);
                    sid::shift(cache, sid::get_stride_element<k_cache_original<Plh>, dim::k>(strides), From());

#pragma unroll
                    for (int i = 0; i < Count::value; ++i) {
                        if (!bound_checker(k_pos))
                            return;
                        if (extent_validator())
                            Sync()(*cache, *original);
                        sid::shift(original, sid::get_stride_element<k_cache_original<Plh>, dim::k>(strides), Step());
                        sid::shift(cache, sid::get_stride_element<Plh, dim::k>(strides), Step());
                        k_pos += Step::value;
                    }
                }
            };

            template <class Plh, class BoundChecker, class Sync, class Offset>
            struct sync_stage {
                template <class Ptrs, class Strides, class ExtentValidator>
                GT_FUNCTION_DEVICE void operator()(Ptrs const &GT_RESTRICT ptrs,
                    Strides const &GT_RESTRICT strides,
                    ExtentValidator &&extent_validator) const {
                    auto k_pos = *device::at_key<positional<dim::k>>(ptrs) + Offset();
                    auto &&bound_checker = *device::at_key<BoundChecker>(ptrs);

                    auto original = device::at_key<k_cache_original<Plh>>(ptrs);
                    sid::shift(original, sid::get_stride_element<k_cache_original<Plh>, dim::k>(strides), Offset());

                    auto cache = device::at_key<Plh>(ptrs);
                    sid::shift(cache, sid::get_stride_element<Plh, dim::k>(strides), Offset());
                    if (bound_checker(k_pos) && extent_validator())
                        Sync()(*cache, *original);
                }
            };

            struct fill {
                template <class Cache, class Original>
                GT_FUNCTION_DEVICE void operator()(Cache &cache, Original const &GT_RESTRICT original) const {
                    cache = original;
                }
            };
            struct flush {
                template <class Cache, class Original>
                GT_FUNCTION_DEVICE void operator()(Cache const &GT_RESTRICT cache, Original &original) const {
                    original = cache;
                }
            };

            template <class>
            struct k_cache_upper_bound_checker {};
            template <class>
            struct k_cache_lower_bound_checker {};

            template <class Sync, class All, class ExecutionType, class Plh>
            using bound_checker =
                meta::if_c<execute::is_forward<ExecutionType>::value == std::is_same<Sync, fill>::value == All::value,
                    k_cache_lower_bound_checker<Plh>,
                    k_cache_upper_bound_checker<Plh>>;

            template <class Sync, class ExecutionType, class Extent>
            using start_offset =
                meta::if_c<execute::is_forward<ExecutionType>::value == std::is_same<Sync, fill>::value,
                    typename Extent::kplus,
                    typename Extent::kminus>;

            template <class Mss,
                class Plh,
                class Sync,
                class All,
                class Extent = extract_k_extent_for_cache<Plh, typename Mss::esf_sequence_t>,
                class BoundsChecker = bound_checker<Sync, All, typename Mss::execution_engine_t, Plh>,
                class StartOffset = start_offset<Sync, typename Mss::execution_engine_t, Extent>>
            struct sync_stage_type;

            template <class Mss, class Plh, class Sync, class Extent, class BoundsChecker, class StartOffset>
            struct sync_stage_type<Mss, Plh, Sync, std::false_type, Extent, BoundsChecker, StartOffset> {
                using type = sync_stage<Plh, BoundsChecker, Sync, StartOffset>;
            };

            template <class Mss, class Plh, class Sync, class Extent, class BoundsChecker, class StartOffset>
            struct sync_stage_type<Mss, Plh, Sync, std::true_type, Extent, BoundsChecker, StartOffset> {
                using type = sync_all_stage<Plh,
                    BoundsChecker,
                    Sync,
                    StartOffset,
                    integral_constant<int_t, Extent::kplus::value - Extent::kminus::value + 1>,
                    integral_constant<int_t, StartOffset::value == Extent::kminus::value ? 1 : -1>>;
            };

            template <class Mss, class Sync, class All>
            struct sync_stage_f {
                template <class Plh>
                using apply = typename sync_stage_type<Mss, Plh, Sync, All>::type;
            };

            template <class Sync>
            using sync_predicate =
                meta::if_<std::is_same<Sync, fill>, meta::curry<is_filling_cache>, meta::curry<is_flushing_cache>>;

            template <class Mss, class Sync>
            using sync_plhs = meta::transform<cache_parameter,
                meta::filter<sync_predicate<Sync>::template apply, typename Mss::cache_sequence_t>>;

            template <class Mss>
            struct add_sync_stages_f {
                template <class Stages,
                    class Sync,
                    class SyncAll,
                    class Plhs = sync_plhs<Mss, Sync>,
                    class SyncStages = meta::transform<sync_stage_f<Mss, Sync, SyncAll>::template apply, Plhs>>
                using apply = meta::
                    if_<std::is_same<Sync, fill>, meta::concat<SyncStages, Stages>, meta::concat<Stages, SyncStages>>;
            };

            template <class T, class = void>
            struct is_constant_one : std::false_type {};

            template <class T>
            struct is_constant_one<T, std::enable_if_t<T::value == 1>> : std::true_type {};

            template <template <class...> class AddStages,
                class Sync,
                class SyncAll,
                class Count,
                class Stages,
                std::enable_if_t<!SyncAll::value || is_constant_one<Count>::value, int> = 0>
            auto add_sync_stages_to_interval(Sync, SyncAll, loop_interval<Count, Stages> interval) {
                return tuple_util::make<tuple>(
                    make_loop_interval(interval.count(), AddStages<Stages, Sync, SyncAll>()));
            }

            template <template <class...> class AddStages,
                class Count,
                class Stages,
                std::enable_if_t<!is_constant_one<Count>::value, int> = 0>
            auto add_sync_stages_to_interval(fill, std::true_type, loop_interval<Count, Stages> interval) {
                assert(interval.count() > 1);
                using namespace literals;
                return tuple_util::make<tuple>(make_loop_interval(1_c, AddStages<Stages, fill, std::true_type>()),
                    make_loop_interval(interval.count() - 1_c, AddStages<Stages, fill, std::false_type>()));
            }

            template <template <class...> class AddStages,
                class Count,
                class Stages,
                std::enable_if_t<!is_constant_one<Count>::value, int> = 0>
            auto add_sync_stages_to_interval(flush, std::true_type, loop_interval<Count, Stages> interval) {
                assert(interval.count() > 1);
                using namespace literals;
                return tuple_util::make<tuple>(
                    make_loop_interval(interval.count() - 1_c, AddStages<Stages, flush, std::false_type>()),
                    make_loop_interval(1_c, AddStages<Stages, flush, std::true_type>()));
            }

            template <class Mss, class Sync>
            using has_caches = meta::any_of<sync_predicate<Sync>::template apply, typename Mss::cache_sequence_t>;

            template <class Mss, class LoopIntervals, std::enable_if_t<has_caches<Mss, fill>::value, int> = 0>
            auto add_sync_stages(fill, LoopIntervals loop_intervals) {
                return tuple_util::flatten(tuple_util::transform(
                    [](auto loop_interval, auto index) {
                        return add_sync_stages_to_interval<add_sync_stages_f<Mss>::template apply>(
                            fill(), bool_constant<decltype(index)::value == 0>(), std::move(loop_interval));
                    },
                    std::move(loop_intervals),
                    meta::make_indices<tuple_util::size<LoopIntervals>, tuple>()));
            }

            template <class Mss, class LoopIntervals, std::enable_if_t<has_caches<Mss, flush>::value, int> = 0>
            auto add_sync_stages(flush, LoopIntervals loop_intervals) {
                using len_t = tuple_util::size<LoopIntervals>;
                return tuple_util::flatten(tuple_util::transform(
                    [](auto loop_interval, auto index) {
                        return add_sync_stages_to_interval<add_sync_stages_f<Mss>::template apply>(flush(),
                            bool_constant<decltype(index)::value + 1 == len_t::value>(),
                            std::move(loop_interval));
                    },
                    std::move(loop_intervals),
                    meta::make_indices<len_t, tuple>()));
            }

            template <class Mss,
                class Sync,
                class LoopIntervals,
                std::enable_if_t<!has_caches<Mss, Sync>::value, int> = 0>
            auto add_sync_stages(Sync, LoopIntervals loop_intervals) {
                return loop_intervals;
            }

            template <class Mss, class LoopIntervals>
            auto add_fill_flush_stages(LoopIntervals loop_intervals) {
                return add_sync_stages<Mss>(flush(), add_sync_stages<Mss>(fill(), std::move(loop_intervals)));
            }

            struct dummy_bound_checker_f {
                GT_FUNCTION_DEVICE bool operator()(int_t) const { return true; }
            };

            template <class Bound>
            struct lower_bound_checker_f : tuple<Bound> {
                lower_bound_checker_f(Bound bound) : tuple<Bound>(bound) {}
                GT_FUNCTION_DEVICE bool operator()(int_t pos) const { return pos >= tuple_util::device::get<0>(*this); }
            };

            template <class Bound>
            struct upper_bound_checker_f : tuple<Bound> {
                upper_bound_checker_f(Bound bound) : tuple<Bound>(bound) {}
                GT_FUNCTION_DEVICE bool operator()(int_t pos) const { return pos < tuple_util::device::get<0>(*this); }
            };

            template <class Extent,
                class Bounds,
                std::enable_if_t<(Extent::kminus::value < 0) && has_key<Bounds, dim::k>::value, int> = 0>
            auto make_lower_bound_checker(Bounds bounds) {
                auto bound = at_key<dim::k>(bounds);
                return lower_bound_checker_f<decltype(bound)>(bound);
            }
            template <class Extent,
                class Bounds,
                std::enable_if_t<(Extent::kminus::value >= 0) || !has_key<Bounds, dim::k>::value, int> = 0>
            dummy_bound_checker_f make_lower_bound_checker(Bounds) {
                return {};
            }

            template <class Extent,
                class Bounds,
                std::enable_if_t<(Extent::kplus::value > 0) && has_key<Bounds, dim::k>::value, int> = 0>
            auto make_upper_bound_checker(Bounds bounds) {
                auto bound = at_key<dim::k>(bounds);
                return upper_bound_checker_f<decltype(bound)>(bound);
            }
            template <class Extent,
                class Bounds,
                std::enable_if_t<(Extent::kplus::value <= 0) || !has_key<Bounds, dim::k>::value, int> = 0>
            dummy_bound_checker_f make_upper_bound_checker(Bounds) {
                return {};
            }

            template <class Mss, class Plhs, class DataStores>
            auto make_lower_bound_sids(DataStores const &data_stores) {
                return tuple_util::transform(
                    [&](auto plh) {
                        using plh_t = decltype(plh);
                        using extent_t = extract_k_extent_for_cache<plh_t, typename Mss::esf_sequence_t>;
                        GT_STATIC_ASSERT((has_key<DataStores, plh_t>::value), GT_INTERNAL_ERROR);
                        auto bounds = sid::get_lower_bounds(at_key<plh_t>(data_stores));
                        return make_global_parameter(make_lower_bound_checker<extent_t>(std::move(bounds)));
                    },
                    hymap::from_keys_values<meta::transform<k_cache_lower_bound_checker, Plhs>, Plhs>());
            }

            template <class Mss, class Plhs, class DataStores>
            auto make_upper_bound_sids(DataStores const &data_stores) {
                return tuple_util::transform(
                    [&](auto plh) {
                        using plh_t = decltype(plh);
                        using extent_t = extract_k_extent_for_cache<plh_t, typename Mss::esf_sequence_t>;
                        GT_STATIC_ASSERT((has_key<DataStores, plh_t>::value), GT_INTERNAL_ERROR);
                        auto bounds = sid::get_upper_bounds(at_key<plh_t>(data_stores));
                        return make_global_parameter(make_upper_bound_checker<extent_t>(std::move(bounds)));
                    },
                    hymap::from_keys_values<meta::transform<k_cache_upper_bound_checker, Plhs>, Plhs>());
            }

            template <class Mss, class DataStores>
            auto make_bound_checkers(DataStores const &data_stores) {
                using plhs_t = meta::dedup<meta::concat<sync_plhs<Mss, fill>, sync_plhs<Mss, flush>>>;
                return hymap::concat(
                    make_lower_bound_sids<Mss, plhs_t>(data_stores), make_upper_bound_sids<Mss, plhs_t>(data_stores));
            }
        } // namespace fill_flush_impl_

        using fill_flush_impl_::add_fill_flush_stages;
        using fill_flush_impl_::k_cache_original;
        using fill_flush_impl_::make_bound_checkers;
    } // namespace cuda
} // namespace gridtools

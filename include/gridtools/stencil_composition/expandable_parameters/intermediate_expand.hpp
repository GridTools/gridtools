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
#include <cassert>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../common/defs.hpp"
#include "../../common/functional.hpp"
#include "../../common/split_args.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../arg.hpp"
#include "../esf_fwd.hpp"
#include "../esf_metafunctions.hpp"
#include "../fused_mss_loop.hpp"
#include "../independent_esf.hpp"
#include "../intermediate.hpp"
#include "../mss.hpp"

namespace gridtools {

    namespace _impl {
        // TODO(anstaf): improve readability of this type computation
        namespace expand_detail {
            // The purpose of this tag is to uniquely identify the args for each index in the unrolling of the
            // expandable parameters (0...ExpandFactor).
            template <size_t I, class Tag>
            struct unrolled_tag;

            /*
             * The logic here is the following:
             *   - `convert_plh` does the actual job;
             *   - the way how it is implemented dictates that it should be lazy.
             *   - so we are placing it into `lazy` namespace and use standard `meta` macro to expose it out.
             *   - that macro supports only plain functions, not the functions that return meta classes
             *     (in `meta` terminology)
             *   - to use it later with `meta::transform` we need to bend `convert_plh<I, Plh>` into a function that
             *     takes just `I` and returns a function that takes `Plh`.
             *   - this is actually `convert_plh_f`. The suffix `_f` is used to stress that it returns a function
             *     (meta class). He the inner name `apply` is not arbitrary. It just the requirement of meta class
             *     concept.
             */
            GT_META_LAZY_NAMESPACE {
                template <size_t I, class Plh>
                struct convert_plh {
                    using type = Plh;
                };

                template <size_t I, class ID, class DataStore, class Location, bool Temporary>
                struct convert_plh<I, plh<ID, std::vector<DataStore>, Location, Temporary>> {
                    using type = plh<unrolled_tag<I, ID>, DataStore, Location, Temporary>;
                };

                template <class I, class Cache>
                struct convert_cache;

                template <class I, cache_type CacheType, class Plh, cache_io_policy cacheIOPolicy>
                struct convert_cache<I, detail::cache_impl<CacheType, Plh, cacheIOPolicy>> {
                    using type =
                        detail::cache_impl<CacheType, typename convert_plh<I::value, Plh>::type, cacheIOPolicy>;
                };

                template <class I, class ArgStoragePair>
                struct convert_arg_storage_pair;

                template <class I, class Plh, class DataStore>
                struct convert_arg_storage_pair<I, arg_storage_pair<Plh, std::vector<DataStore>>> {
                    using type = arg_storage_pair<typename convert_plh<I::value, Plh>::type, DataStore>;
                };
            };
            GT_META_DELEGATE_TO_LAZY(convert_plh, (size_t I, class Plh), (I, Plh));
            GT_META_DELEGATE_TO_LAZY(convert_cache, (class I, class Cache), (I, Cache));
            GT_META_DELEGATE_TO_LAZY(convert_arg_storage_pair, (class I, class ArgStoragePair), (I, ArgStoragePair));

            template <class IndexAndCache>
            GT_META_DEFINE_ALIAS(convert_cache_f,
                convert_cache,
                (GT_META_CALL(meta::first, IndexAndCache), GT_META_CALL(meta::second, IndexAndCache)));

            template <size_t I>
            struct convert_plh_f {
                template <class Plh>
                GT_META_DEFINE_ALIAS(apply, convert_plh, (I, Plh));
            };

            template <class ArgStoragePair>
            struct convert_arg_storage_pair_f {
                template <class I>
                GT_META_DEFINE_ALIAS(apply, convert_arg_storage_pair, (I, ArgStoragePair));
            };

            template <size_t ExpandFactor,
                class Caches,
                class Indices = GT_META_CALL(meta::make_indices_c, ExpandFactor),
                class IndicesAndCaches = GT_META_CALL(meta::cartesian_product, (Indices, Caches)),
                class RawExpandedCaches = GT_META_CALL(meta::transform, (convert_cache_f, IndicesAndCaches)),
                class ExpandedCaches = GT_META_CALL(meta::dedup, RawExpandedCaches)>
            GT_META_DEFINE_ALIAS(expand_caches, meta::rename, (meta::ctor<std::tuple<>>::apply, ExpandedCaches));

            template <size_t ExpandFactor, class ArgStoragePair>
            GT_META_DEFINE_ALIAS(expand_arg_storage_pair,
                meta::rename,
                (meta::ctor<std::tuple<>>::apply,
                    GT_META_CALL(meta::transform,
                        (convert_arg_storage_pair_f<ArgStoragePair>::template apply,
                            GT_META_CALL(meta::make_indices_c, ExpandFactor)))));

            template <class I>
            struct data_store_generator_f {
                template <class DataStore>
                DataStore const &operator()(std::vector<DataStore> const &src, size_t offset) const {
                    assert(offset + I::value < src.size());
                    return src[offset + I::value];
                }
                using type = data_store_generator_f;
            };

            template <size_t ExpandFactor>
            struct expand_arg_storage_pair_f {
                size_t m_offset;
                template <class T, class Res = GT_META_CALL(expand_arg_storage_pair, (ExpandFactor, T))>
                Res operator()(T const &obj) const {
                    using indices_t = GT_META_CALL(meta::make_indices_c, ExpandFactor);
                    using generators_t = GT_META_CALL(meta::transform, (data_store_generator_f, indices_t));
                    auto const &src = obj.m_value;
                    Res res = tuple_util::generate<generators_t, Res>(src, m_offset);
                    return res;
                }
            };

            template <size_t I, class Plhs>
            GT_META_DEFINE_ALIAS(convert_plhs, meta::transform, (convert_plh_f<I>::template apply, Plhs));

            template <class Esf>
            struct convert_esf {
                template <class I>
                GT_META_DEFINE_ALIAS(
                    apply, esf_replace_args, (Esf, GT_META_CALL(convert_plhs, (I::value, typename Esf::args_t))));
            };

            template <size_t ExpandFactor>
            struct expand_normal_esf_f {
                template <class Esf>
                GT_META_DEFINE_ALIAS(apply,
                    meta::transform,
                    (convert_esf<Esf>::template apply, GT_META_CALL(meta::make_indices_c, ExpandFactor)));
            };

            template <size_t ExpandFactor>
            struct expand_esf_f;

            GT_META_LAZY_NAMESPACE {
                template <size_t ExpandFactor, class Esf>
                struct expand_esf {
                    using indices_t = GT_META_CALL(meta::make_indices_c, ExpandFactor);
                    using esfs_t = GT_META_CALL(meta::transform, (convert_esf<Esf>::template apply, indices_t));
                    using tuple_t = GT_META_CALL(meta::rename, (meta::ctor<std::tuple<>>::apply, esfs_t));
                    using type = independent_esf<tuple_t>;
                };
                template <size_t ExpandFactor, class Esfs>
                struct expand_esf<ExpandFactor, independent_esf<Esfs>> {
                    using type = independent_esf<GT_META_CALL(meta::flatten,
                        (GT_META_CALL(meta::transform, (expand_normal_esf_f<ExpandFactor>::template apply, Esfs))))>;
                };

                template <size_t ExpandFactor, class Mss>
                struct convert_mss;

                template <size_t ExpandFactor, class ExecutionEngine, class Esfs, class Caches>
                struct convert_mss<ExpandFactor, mss_descriptor<ExecutionEngine, Esfs, Caches>> {
                    using esfs_t = GT_META_CALL(meta::transform, (expand_esf_f<ExpandFactor>::template apply, Esfs));
                    using type =
                        mss_descriptor<ExecutionEngine, esfs_t, GT_META_CALL(expand_caches, (ExpandFactor, Caches))>;
                };
            }
            GT_META_DELEGATE_TO_LAZY(expand_esf, (size_t ExpandFactor, class Esf), (ExpandFactor, Esf));
            GT_META_DELEGATE_TO_LAZY(convert_mss, (size_t ExpandFactor, class Mss), (ExpandFactor, Mss));

            template <size_t ExpandFactor>
            struct expand_esf_f {
                template <class Esf>
                GT_META_DEFINE_ALIAS(apply, expand_esf, (ExpandFactor, Esf));
            };

            template <size_t ExpandFactor>
            struct convert_mss_descriptor_f {
                template <class T>
                GT_META_DEFINE_ALIAS(apply, convert_mss, (ExpandFactor, T));
            };

            template <typename T>
            struct is_expandable : std::false_type {};

            template <typename Arg, typename DataStore>
            struct is_expandable<arg_storage_pair<Arg, std::vector<DataStore>>> : std::true_type {};

            struct get_value_size {
                template <class T>
                size_t operator()(T const &t) const {
                    return t.m_value.size();
                }
#ifndef BOOST_RESULT_OF_USE_DECLTYPE
                using result_type = size_t;
#endif
            };

            template <typename ArgStoragePairs>
            enable_if_t<!meta::is_empty<ArgStoragePairs>::value, size_t> get_expandable_size(
                ArgStoragePairs const &src) {
                auto sizes = tuple_util::transform(get_value_size{}, src);
                size_t res = tuple_util::get<0>(sizes);
                assert(tuple_util::fold([res](bool left, size_t size) { return left && size == res; })(true, sizes));
                return res;
            }

            template <typename ArgStoragePairs>
            enable_if_t<meta::is_empty<ArgStoragePairs>::value, size_t> get_expandable_size(
                ArgStoragePairs const &src) {
                // If there is nothing to expand we are going to compute stencils once.
                return 1;
            }

            template <uint_t ExpandFactor, class ArgStoragePairs>
            auto convert_arg_storage_pairs(size_t offset, ArgStoragePairs const &src)
                GT_AUTO_RETURN(tuple_util::deep_copy(
                    tuple_util::flatten(tuple_util::transform(expand_arg_storage_pair_f<ExpandFactor>{offset}, src))));

            template <uint_t ExpandFactor, class MssDescriptors>
            GT_META_DEFINE_ALIAS(converted_mss_descriptors,
                meta::transform,
                (convert_mss_descriptor_f<ExpandFactor>::template apply, MssDescriptors));

            template <class Intermediate>
            struct run_f {
                Intermediate &m_intermediate;
                template <class... Args>
                void operator()(Args const &... args) const {
                    m_intermediate.run(args...);
                }
                using result_type = void;
            };

            template <class Intermediate, class Args>
            void invoke_run(Intermediate &intermediate, Args &&args) {
                tuple_util::apply(run_f<Intermediate>{intermediate}, wstd::forward<Args>(args));
            }
        } // namespace expand_detail
    }     // namespace _impl
    /**
     * @file
     * \brief this file contains the intermediate representation used in case of expandable parameters
     * */

    /**
       @brief the intermediate representation object

       The expandable parameters are long lists of storages on which the same stencils are applied,
       in a Single-Stencil-Multiple-Storage way. In order to avoid resource contention usually
       it is convenient to split the execution in multiple stencil, each stencil operating on a chunk
       of the list. Say that we have an expandable parameters list of length 23, and a chunk size of
       4, we'll execute 5 stencil with a "vector width" of 4, and one stencil with a "vector width"
       of 3 (23%4).

       This object contains two objects of @ref gridtools::intermediate type, one with a vector width
       corresponding to the expand factor defined by the user (4 in the previous example), and another
       one with a vector width of expand_factor%total_parameters (3 in the previous example).
     */
    template <size_t ExpandFactor,
        bool IsStateful,
        class Backend,
        class Grid,
        class BoundArgStoragePairs,
        class MssDescriptors>
    class intermediate_expand {
        using non_expandable_bound_arg_storage_pairs_t = GT_META_CALL(
            meta::filter, (meta::not_<_impl::expand_detail::is_expandable>::apply, BoundArgStoragePairs));
        using expandable_bound_arg_storage_pairs_t = GT_META_CALL(
            meta::filter, (_impl::expand_detail::is_expandable, BoundArgStoragePairs));

        template <size_t N>
        using converted_intermediate = intermediate<IsStateful,
            Backend,
            Grid,
            non_expandable_bound_arg_storage_pairs_t,
            GT_META_CALL(_impl::expand_detail::converted_mss_descriptors, (N, MssDescriptors))>;

        /// Storages that are expandable, is bound in construction time.
        //
        expandable_bound_arg_storage_pairs_t m_expandable_bound_arg_storage_pairs;

        /// The object of `intermediate` type to which the computation will be delegated.
        //
        converted_intermediate<ExpandFactor> m_intermediate;

        /// If the actual size of storages is not divided by `ExpandFactor`, this `intermediate` will process
        /// reminder.
        converted_intermediate<1> m_intermediate_remainder;

        typename timer_traits<Backend>::timer_type m_meter;

        template <class ExpandableBoundArgStoragePairRefs, class NonExpandableBoundArgStoragePairRefs>
        intermediate_expand(Grid const &grid,
            std::pair<ExpandableBoundArgStoragePairRefs, NonExpandableBoundArgStoragePairRefs> &&arg_refs)
            // expandable arg_storage_pairs are kept as a class member until run will be called.
            : m_expandable_bound_arg_storage_pairs(wstd::move(arg_refs.first)),
              // plain arg_storage_pairs are bound to both intermediates;
              m_intermediate(grid, arg_refs.second, false), m_intermediate_remainder(grid, arg_refs.second, false),
              m_meter("NoName") {}

      public:
        template <class BoundArgStoragePairsRefs>
        intermediate_expand(Grid const &grid, BoundArgStoragePairsRefs &&arg_storage_pairs)
            // public constructor splits given ard_storage_pairs to expandable and plain ones and delegates to the
            // private constructor.
            : intermediate_expand(
                  grid, split_args_tuple<_impl::expand_detail::is_expandable>(wstd::move(arg_storage_pairs))) {}

        template <class... Args, class... DataStores>
        void run(arg_storage_pair<Args, DataStores> const &... args) {
            m_meter.start();
            // split arguments to expandable and plain arg_storage_pairs
            auto arg_groups = split_args<_impl::expand_detail::is_expandable>(args...);
            auto bound_expandable_arg_refs = tuple_util::transform(identity{}, m_expandable_bound_arg_storage_pairs);
            // concatenate expandable portion of arguments with the refs to bound expandable ard_storage_pairs
            auto expandable_args =
                std::tuple_cat(wstd::move(bound_expandable_arg_refs), wstd::move(arg_groups.first));
            const auto &plain_args = arg_groups.second;
            // extract size from the vectors within expandable args.
            // if vectors are not of the same length assert within `get_expandable_size` fails.
            size_t size = _impl::expand_detail::get_expandable_size(expandable_args);
            size_t offset = 0;
            for (; size - offset >= ExpandFactor; offset += ExpandFactor) {
                // form the chunks from expandable_args with the given offset
                auto converted_args =
                    _impl::expand_detail::convert_arg_storage_pairs<ExpandFactor>(offset, expandable_args);
                // concatenate that chunk with the plain portion of the arguments
                // and invoke the `run` of the `m_intermediate`.
                _impl::expand_detail::invoke_run(
                    m_intermediate, tuple_util::flatten(std::tie(plain_args, converted_args)));
            }
            // process the reminder the same way
            for (; offset < size; ++offset) {
                auto converted_args = _impl::expand_detail::convert_arg_storage_pairs<1>(offset, expandable_args);
                _impl::expand_detail::invoke_run(
                    m_intermediate_remainder, tuple_util::flatten(std::tie(plain_args, converted_args)));
            }
            m_meter.pause();
        }

        std::string print_meter() const { return m_meter.to_string(); }

        double get_time() const { return m_meter.total_time(); }

        size_t get_count() const { return m_meter.count(); }

        void reset_meter() { m_meter.reset(); }

        template <class Placeholder>
        static constexpr auto get_arg_extent(Placeholder) GT_AUTO_RETURN(converted_intermediate<1>::get_arg_extent(
            GT_META_CALL(_impl::expand_detail::convert_plh, (0, Placeholder)){}));
        template <class Placeholder>
        static constexpr auto get_arg_intent(Placeholder) GT_AUTO_RETURN(converted_intermediate<1>::get_arg_intent(
            GT_META_CALL(_impl::expand_detail::convert_plh, (0, Placeholder)){}));
    };
} // namespace gridtools

/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#pragma once
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <functional>
#include <vector>
#include <utility>
#include <memory>

#include <boost/mpl/logical.hpp>
#include <boost/mpl/transform.hpp>

#include <boost/fusion/include/any.hpp>
#include <boost/fusion/include/count.hpp>
#include <boost/fusion/include/empty.hpp>
#include <boost/fusion/include/filter_if.hpp>
#include <boost/fusion/include/filter_view.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/front.hpp>
#include <boost/fusion/include/invoke.hpp>
#include <boost/fusion/include/make_fused.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/fusion/include/transform.hpp>
#include <boost/fusion/include/zip_view.hpp>

#include "../../common/defs.hpp"
#include "../../common/vector_traits.hpp"
#include "../../common/functional.hpp"

#include "../../common/split_args.hpp"
#include "../../common/tuple_util.hpp"
#include "../../common/generic_metafunctions/meta.hpp"

#include "../../storage/data_store_field.hpp"
#include "../arg.hpp"
#include "../backend_metafunctions.hpp"
#include "../computation_grammar.hpp"
#include "../grid.hpp"
#include "../intermediate.hpp"
#include "../intermediate_impl.hpp"
#include "../mss_components_metafunctions.hpp"
#include "../conditionals/condition_tree.hpp"
#include "expand_factor.hpp"

namespace gridtools {

    namespace _impl {
        namespace expand_detail {
            template < typename T >
            struct is_expandable : std::false_type {};

            template < typename Arg, typename DataStoreType >
            struct is_expandable< arg_storage_pair< Arg, DataStoreType > > : is_vector< DataStoreType > {};

            template < uint_t N, typename T >
            struct convert_data_store_type {
                using type = T;
            };
            template < uint_t N, typename T >
            struct convert_data_store_type< N, std::vector< T > > {
                using type = data_store_field< T, N >;
            };

            template < uint_t N >
            struct convert_placeholder {
                template < typename >
                struct apply;
                template < uint_t I, typename S, typename L, bool T >
                struct apply< arg< I, S, L, T > > {
                    using type = arg< I, typename convert_data_store_type< N, S >::type, L, T >;
                };
            };

            struct get_value_size {
                template < class T >
                size_t operator()(T const &t) const {
                    return t.m_value.size();
                }
#ifndef BOOST_RESULT_OF_USE_DECLTYPE
                using result_type = size_t;
#endif
            };

            template < typename ArgStoragePairs >
            typename std::enable_if< !boost::mpl::empty< ArgStoragePairs >::value, size_t >::type get_expandable_size(
                ArgStoragePairs const &src) {
                namespace f = boost::fusion;
                auto sizes = f::transform(src, get_value_size{});
                size_t res = f::front(sizes);
                assert(f::any(sizes, [=](size_t size) { return size == res; }));
                return res;
            }

            template < typename ArgStoragePairs >
            typename std::enable_if< boost::mpl::empty< ArgStoragePairs >::value, size_t >::type get_expandable_size(
                ArgStoragePairs const &src) {
                // If there is nothing to expand we are going to compute stensil once.
                return 1;
            }

            template < uint_t N >
            struct convert_data_store_f {
                size_t m_offset;
                template < typename T >
                data_store_field< T, N > operator()(const std::vector< T > &src) const {
                    assert(!src.empty());
                    data_store_field< T, N > res = {*src[0].get_storage_info_ptr()};
                    assert(src.size() >= m_offset + N);
                    for (uint_t i = 0; i != N; ++i)
                        res.set(0, i, src[m_offset + i]);
                    return res;
                }
            };

            template < uint_t N >
            struct convert_arg_storage_pair_f {
                convert_data_store_f< N > m_convert_data_store;
                template < typename Arg, typename DataStoreType >
                arg_storage_pair< typename convert_placeholder< N >::template apply< Arg >::type,
                    typename convert_data_store_type< N, DataStoreType >::type >
                operator()(arg_storage_pair< Arg, DataStoreType > const &src) const {
                    return {m_convert_data_store(src.m_value)};
                }
            };

            template < uint_t N, class ArgStoragePairs >
            auto convert_arg_storage_pairs(size_t offset, ArgStoragePairs const &src)
                GT_AUTO_RETURN(tuple_util::transform(convert_arg_storage_pair_f< N >{offset}, src));

            template < uint_t N >
            struct convert_mss_descriptors_tree_f {
                template < typename T >
                auto operator()(T const &src) const
                    GT_AUTO_RETURN(condition_tree_transform(src, fix_mss_arg_indices_f< N >{}));
            };

            template < uint_t N, class... Ts >
            auto convert_mss_descriptors_trees(std::tuple< Ts... > const &src)
                GT_AUTO_RETURN(tuple_util::transform(convert_mss_descriptors_tree_f< N >{}, src));

            template < uint_t N, typename MssDescriptorsTrees >
            using converted_mss_descriptors_trees =
                decltype(convert_mss_descriptors_trees< N >(std::declval< MssDescriptorsTrees const & >()));

            template < class Intermediate >
            struct run_f {
                Intermediate &m_intermediate;
                template < class... Args >
                void operator()(Args const &... args) const {
                    m_intermediate.run(args...);
                }
                using result_type = void;
            };

            template < class Intermediate, class Args >
            void invoke_run(Intermediate &intermediate, Args &&args) {
                boost::fusion::invoke(run_f< Intermediate >{intermediate}, std::forward< Args >(args));
            }

            struct sync_f {
                template < class Arg, class DataStore >
                void operator()(arg_storage_pair< Arg, std::vector< DataStore > > const &obj) const {
                    for (auto &&item : obj.m_value)
                        item.sync();
                }
            };
        }
    }
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
    template < uint_t ExpandFactor,
        bool IsStateful,
        class Backend,
        class Grid,
        class BoundArgStoragePairs,
        class MssDescriptorTrees >
    class intermediate_expand {
        using non_expandable_bound_arg_storage_pairs_t = GT_META_CALL(
            meta::filter, (meta::not_< _impl::expand_detail::is_expandable >::apply, BoundArgStoragePairs));
        using expandable_bound_arg_storage_pairs_t = GT_META_CALL(
            meta::filter, (_impl::expand_detail::is_expandable, BoundArgStoragePairs));

        template < uint_t N >
        using converted_intermediate = intermediate< N,
            IsStateful,
            Backend,
            Grid,
            non_expandable_bound_arg_storage_pairs_t,
            _impl::expand_detail::converted_mss_descriptors_trees< N, MssDescriptorTrees > >;

        /// Storages that are expandable, is bound in construction time.
        //
        expandable_bound_arg_storage_pairs_t m_expandable_bound_arg_storage_pairs;

        /// The object of `intermediate` type to which the computation will be delegated.
        //
        converted_intermediate< ExpandFactor > m_intermediate;

        /// If the actual size of storages is not divided by `ExpandFactor`, this `intermediate` will process
        /// reminder.
        converted_intermediate< 1 > m_intermediate_remainder;

        typedef typename Backend::backend_traits_t::performance_meter_t performance_meter_t;
        performance_meter_t m_meter;

        template < class ExpandableBoundArgStoragePairRefs, class NonExpandableBoundArgStoragePairRefs >
        intermediate_expand(Grid const &grid,
            std::pair< ExpandableBoundArgStoragePairRefs, NonExpandableBoundArgStoragePairRefs > &&arg_refs,
            MssDescriptorTrees const &msses)
            // expandable arg_storage_pairs are kept as a class member until run will be called.
            : m_expandable_bound_arg_storage_pairs(std::move(arg_refs.first)),
              // plain arg_storage_pairs are bound to both intermediates;
              // msses descriptors got transformed and also got passed to intermediates.
              m_intermediate(
                  grid, arg_refs.second, _impl::expand_detail::convert_mss_descriptors_trees< ExpandFactor >(msses)),
              m_intermediate_remainder(
                  grid, arg_refs.second, _impl::expand_detail::convert_mss_descriptors_trees< 1 >(msses)),
              m_meter("NoName") {}

      public:
        template < class BoundArgStoragePairsRefs >
        intermediate_expand(
            Grid const &grid, BoundArgStoragePairsRefs &&arg_storage_pairs, MssDescriptorTrees const &msses)
            // public constructor splits given ard_storage_pairs to expandable and plain ones and delegates to the
            // private constructor.
            : intermediate_expand(
                  grid, split_args_tuple< _impl::expand_detail::is_expandable >(std::move(arg_storage_pairs)), msses) {}

        template < class... Args, class... DataStores >
        notype run(arg_storage_pair< Args, DataStores > const &... args) {
            // split arguments to expadable and plain ard_storage_pairs
            auto arg_groups = split_args< _impl::expand_detail::is_expandable >(args...);
            auto bound_expandable_arg_refs = tuple_util::transform(identity{}, m_expandable_bound_arg_storage_pairs);
            // concatenate expandable portion of arguments with the refs to bound expandable ard_storage_pairs
            auto expandable_args = std::tuple_cat(std::move(bound_expandable_arg_refs), std::move(arg_groups.first));
            const auto &plain_args = arg_groups.second;
            // extract size from the vectors within expandable args.
            // if vectors are not of the same length assert within `get_expandable_size` fails.
            size_t size = _impl::expand_detail::get_expandable_size(expandable_args);
            size_t offset = 0;
            m_meter.start();
            for (; size - offset >= ExpandFactor; offset += ExpandFactor) {
                // form the chunks from expandable_args with the given offset
                auto converted_args =
                    _impl::expand_detail::convert_arg_storage_pairs< ExpandFactor >(offset, expandable_args);
                // concatenate that chunk with the plain portion of the arguments
                // and invoke the `run` of the `m_intermediate`.
                _impl::expand_detail::invoke_run(m_intermediate, std::tuple_cat(plain_args, converted_args));
            }
            // process the reminder the same way
            for (; offset < size; ++offset) {
                auto converted_args = _impl::expand_detail::convert_arg_storage_pairs< 1 >(offset, expandable_args);
                _impl::expand_detail::invoke_run(m_intermediate_remainder, std::tuple_cat(plain_args, converted_args));
            }
            m_meter.pause();
            return {};
        }

        void sync_bound_data_stores() const {
            tuple_util::for_each(_impl::expand_detail::sync_f{}, m_expandable_bound_arg_storage_pairs);
            m_intermediate.sync_bound_data_stores();
            m_intermediate_remainder.sync_bound_data_stores();
        }

        std::string print_meter() const { return m_meter.to_string(); }

        double get_meter() const { return m_meter.total_time(); }

        void reset_meter() { m_meter.reset(); }
    };
}

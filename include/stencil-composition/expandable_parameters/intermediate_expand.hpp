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
#include "../../common/fusion.hpp"

#include "../../common/split_args.hpp"
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

            template < class T >
            using is_expandable_decayed = is_expandable< meta::t_< std::decay< T > > >;

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

            template < uint_t N, typename Placeholders >
            using converted_placeholders =
                typename boost::mpl::transform< Placeholders, convert_placeholder< N > >::type;

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
            size_t get_expandable_size(ArgStoragePairs const &src) {
                namespace f = boost::fusion;
                namespace m = boost::mpl;
                auto sizes = f::transform(f::filter_if< is_expandable< m::_ > >(src), get_value_size{});
                if (f::empty(sizes))
                    // If there is nothing to expand we are going to compute stensil once.
                    return 1;
                size_t res = f::front(sizes);
                assert(
                    f::count(sizes, res) == f::size(sizes) && "Non-tmp expandable parameters must have the same size");
                return res;
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
#ifndef BOOST_RESULT_OF_USE_DECLTYPE
                template < typename >
                struct result;
                template < typename Arg, typename DataStoreType >
                struct result< convert_arg_storage_pair_f(arg_storage_pair< Arg, DataStoreType > const &) > {
                    using type = arg_storage_pair< typename convert_placeholder< N >::template apply< Arg >::type,
                        typename convert_data_store_type< N, DataStoreType >::type >;
                };
#endif
            };

            template < uint_t N, class ArgStoragePairs >
            auto convert_arg_storage_pairs(size_t offset, ArgStoragePairs &src)
                GT_AUTO_RETURN(boost::fusion::transform(src, convert_arg_storage_pair_f< N >{offset}));

            template < uint_t N >
            struct convert_mss_descriptors_tree_f {
                template < typename T >
                auto operator()(T const &src) const
                    GT_AUTO_RETURN(condition_tree_transform(src, fix_mss_arg_indices_f< N >{}));
            };

            template < uint_t N >
            struct convert_mss_descriptors_trees_f {
                template < class... Ts >
                auto operator()(std::tuple< Ts... > const &src) const
                    GT_AUTO_RETURN(as_std_tuple(boost::fusion::transform(src, convert_mss_descriptors_tree_f< N >{})));
            };

            template < uint_t N, typename MssDescriptorsTree >
            auto convert_mss_descriptors_tree(MssDescriptorsTree const &src)
                GT_AUTO_RETURN(convert_mss_descriptors_tree_f< N >{}(src));

            template < uint_t N, typename MssDescriptorsTrees >
            using converted_mss_descriptors_trees =
                decltype(convert_mss_descriptors_trees_f< N >{}(std::declval< MssDescriptorsTrees const & >()));

            template < class Intermediate >
            struct run_f {
                Intermediate &m_intermediate;
                template < class... Args >
                auto operator()(Args const &... args) const GT_AUTO_RETURN(m_intermediate.run(args...));
            };

            template < class Intermediate, class Args >
            auto invoke_run(Intermediate &intermediate, Args &&args)
                GT_AUTO_RETURN(boost::fusion::invoke(run_f< Intermediate >{intermediate}, args));

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
       in a Single-Stencil-Multiple-Storage way. In order to avoid resource contemption usually
       it is convenient to split the execution in multiple stencil, each stencil operating on a chunk
       of the list. Say that we have an expandable parameters list of length 23, and a chunk size of
       4, we'll execute 5 stencil with a "vector width" of 4, and one stencil with a "vector width"
       of 3 (23%4).

       This object contains two unique pointers of @ref gridtools::intermediate type, one with a vector width
       corresponding to the expand factor defined by the user (4 in the previous example), and another
       one with a vector width of expand_factor%total_parameters (3 in the previous example).
       In case the total number of parameters is a multiple of the expand factor, the second
       intermediate object does not get instantiated.
     */
    template < uint_t ExpandFactor,
        bool IsStateful,
        class Backend,
        class Grid,
        class BoundArgStoragePairs,
        class MssDescriptorTrees >
    class intermediate_expand {
        using non_expandable_bound_arg_storage_pairs_t =
            meta::apply< meta::filter< meta::not_< _impl::expand_detail::is_expandable >::template apply >,
                BoundArgStoragePairs >;
        using expandable_bound_arg_storage_pairs_t =
            meta::apply< meta::filter< _impl::expand_detail::is_expandable >, BoundArgStoragePairs >;

        template < uint_t N >
        using converted_intermediate = intermediate< N,
            IsStateful,
            Backend,
            Grid,
            non_expandable_bound_arg_storage_pairs_t,
            _impl::expand_detail::converted_mss_descriptors_trees< N, MssDescriptorTrees > >;

        expandable_bound_arg_storage_pairs_t m_expandable_bound_arg_storage_pairs;
        converted_intermediate< ExpandFactor > m_intermediate;
        converted_intermediate< 1 > m_intermediate_remainder;

        template < class ExpandableBoundArgStoragePairRefs, class NonExpandableBoundArgStoragePairRefs >
        intermediate_expand(Grid const &grid,
            std::pair< ExpandableBoundArgStoragePairRefs, NonExpandableBoundArgStoragePairRefs > &&arg_refs,
            MssDescriptorTrees const &msses)
            : m_expandable_bound_arg_storage_pairs{std::move(arg_refs.first)},
              m_intermediate{grid,
                  arg_refs.second,
                  _impl::expand_detail::convert_mss_descriptors_trees_f< ExpandFactor >{}(msses)},
              m_intermediate_remainder{
                  grid, arg_refs.second, _impl::expand_detail::convert_mss_descriptors_trees_f< 1 >{}(msses)} {}

      public:
        template < class BoundArgStoragePairsRefs >
        intermediate_expand(
            Grid const &grid, BoundArgStoragePairsRefs &&arg_storage_pairs, MssDescriptorTrees const &msses)
            : intermediate_expand(grid,
                  split_args_tuple< _impl::expand_detail::is_expandable_decayed >(std::move(arg_storage_pairs)),
                  msses) {}

        template < class... Args, class... DataStores >
        notype run(arg_storage_pair< Args, DataStores > const &... args) {
            auto arg_groups = split_args< _impl::expand_detail::is_expandable_decayed >(args...);
            auto expandable_args = make_joint_view(m_expandable_bound_arg_storage_pairs, arg_groups.first);
            const auto &plain_args = arg_groups.second;
            size_t size = _impl::expand_detail::get_expandable_size(expandable_args);
            size_t offset = 0;
            for (; size - offset >= ExpandFactor; offset += ExpandFactor) {
                auto converted_args =
                    _impl::expand_detail::convert_arg_storage_pairs< ExpandFactor >(offset, expandable_args);
                _impl::expand_detail::invoke_run(m_intermediate, make_joint_view(plain_args, converted_args));
            }
            for (; offset < size; ++offset) {
                auto converted_args =
                    _impl::expand_detail::convert_arg_storage_pairs< ExpandFactor >(offset, expandable_args);
                _impl::expand_detail::invoke_run(m_intermediate_remainder, make_joint_view(plain_args, converted_args));
            }
            return {};
        }

        void sync_all() {
            boost::fusion::for_each(m_expandable_bound_arg_storage_pairs, _impl::expand_detail::sync_f{});
            m_intermediate.sync_all();
            m_intermediate_remainder.sync_all();
        }

        std::string print_meter() {
            assert(false);
            return {};
        }

        double get_meter() { return m_intermediate.get_meter() + m_intermediate_remainder.get_meter(); }

        void reset_meter() {
            m_intermediate.reset_meter();
            m_intermediate_remainder.reset_meter();
        }
    };

#if 0
    template < uint_t ExpandFactor, bool IsStateful, typename Backend, typename Grid, typename... MssDescriptorTrees >
    class intermediate_expand {
        GRIDTOOLS_STATIC_ASSERT((is_backend< Backend >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::and_< std::true_type,
                                    is_condition_tree_of< MssDescriptorTrees, is_computation_token >... >::value),
            "make_computation args should be mss descriptors or condition trees of mss descriptors");

        using all_mss_descriptors_t = typename branch_selector< MssDescriptorTrees... >::all_leaves_t;
        using placeholders_t = typename extract_placeholders< all_mss_descriptors_t >::type;
        using Aggregator = aggregator_type< placeholders_t >;

        template < uint_t N >
        using converted_intermediate = intermediate< N,
            IsStateful,
            Backend,
            Grid,
            _impl::expand_detail::converted_mss_descriptors_tree< N, MssDescriptorTrees >... >;

        using arg_storage_pair_fusion_list_t = typename Aggregator::arg_storage_pair_fusion_list_t;

        // private members

        arg_storage_pair_fusion_list_t m_arg_storage_pairs;
        const size_t m_size;
        std::unique_ptr< converted_intermediate< ExpandFactor > > m_intermediate;
        // For some reason nvcc goes nuts here (even though the previous line is OK):
        // std::unique_ptr< converted_intermediate< 1 > > m_intermediate_remainder;
        // I have to expand `converted_intermediate` alias manually:
        std::unique_ptr< intermediate< 1,
            IsStateful,
            Backend,
            Grid,
            _impl::expand_detail::converted_mss_descriptors_tree< 1, MssDescriptorTrees >... > >
            m_intermediate_remainder;

      public:
        /**
           @brief constructor

           Given expandable parameters with size N, creates other @ref gristools::expandable_parameters storages with
           dimension given by  @ref gridtools::expand_factor
         */
        intermediate_expand(
            Grid const &grid, Aggregator const &domain, MssDescriptorTrees const &... mss_descriptor_trees)
            : m_arg_storage_pairs(domain.get_arg_storage_pairs()),
              m_size(_impl::expand_detail::get_expandable_size(m_arg_storage_pairs)),
              m_intermediate(m_size >= ExpandFactor
                                 ? create_intermediate< ExpandFactor >(grid, domain, mss_descriptor_trees...)
                                 : nullptr),
              m_intermediate_remainder(m_size % ExpandFactor
                                           ? create_intermediate< 1 >(grid, domain, mss_descriptor_trees...)
                                           : nullptr) {}

        /**
           @brief run the execution

           This method performs a run for the computation on each chunck of expandable parameters.
           Between two iterations it updates the @ref gridtools::aggregator_type, so that the storage
           pointers for the current chunck get substituted by the next chunk. At the end of the
           iterations, if the number of parameters is not multiple of the expand factor, the remaining
           chunck of storage pointers is consumed.
         */
        template < class... Args, class... DataStores >
        notype run(arg_storage_pair< Args, DataStores > const &... args) {
            size_t i = 0;
            for (; m_size - i >= ExpandFactor; i += ExpandFactor) {
                // convert
                assign(*m_intermediate, i);
                m_intermediate->run();
            }
            for (; i < m_size; ++i) {
                assign(*m_intermediate_remainder, i);
                m_intermediate_remainder->run();
            }
            return {};
        }

        /**
           @brief forwards to the m_intermediate member

           does not take into account the remainder kernel executed when the number of parameters is
           not multiple of the expand factor
         */
        std::string print_meter() {
            assert(false);
            return {};
        }

        /**
           @brief forwards to the m_intermediate and m_intermediate_remainder members
         */
        void reset_meter() {
            if (m_intermediate)
                m_intermediate->reset_meter();
            if (m_intermediate_remainder)
                m_intermediate_remainder->reset_meter();
        }

        double get_meter() {
            double res = 0;
            if (m_intermediate)
                res += m_intermediate->get_meter();
            if (m_intermediate_remainder)
                res += m_intermediate_remainder->get_meter();
            return res;
        }

        void sync_all() {
            // sync all data stores
            boost::fusion::for_each(m_arg_storage_pairs, _impl::sync_data_stores());
        }

      private:
        template < uint_t N, typename Res = converted_intermediate< N > >
        static Res *create_intermediate(
            Grid const &grid, const Aggregator &domain, MssDescriptorTrees const &... mss_descriptor_trees) {
            return new Res(grid,
                _impl::expand_detail::convert_aggregator< N >(domain),
                _impl::expand_detail::convert_mss_descriptors_tree< N >(mss_descriptor_trees)...);
        }

        template < typename Dst >
        void assign(Dst &dst, size_t offset) const {
            _impl::expand_detail::assign(m_arg_storage_pairs, dst.get_arg_storage_pairs(), offset);
        }
    };
#endif
}

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

/**
 * @file
 * Utilities for dealing with binary trees nodes are gridtools::condition instantiations.
 *
 * Examples of the tree types in the context of this file:
 *   - [some type, that is not gridstool::condition instantiation] - a tree with the only leaf.
 *   - condition<Mss1, Mss2, Cond> - a tree with one node and two leafs.
 *   - condition<Mss1, condition<Mss2, Mss3, Cond2>, Cond1> - a tree with two nodes and three leafs.
 *
 * In the context of stencil computation condition trees correspond to [possibly nested] `if_` constructs or to
 * `switch/case` construct. I.e. the variadic pack within `make_computation` template function signature in it's
 * general form is a sequence of condition trees of computation tokens. Where computation token is either MSS
 * descriptor or reduction descriptor.
 *
 * This module provides the interface for condition tree manipulations to the rest of stencil computation code base.
 * This interface consists of:
 *    - `is_condition_tree_of` compile time predicate;
 *    - `condition_tree_transform` template function;
 *    - `branch_selector` class.
 *
 *   THe current usage:
 *
 *  - `is_condition_tree_of` is used within static_asserts to protect a user against `make_computation` grammar misuse.
 *  - `condition_tree_transform` is used in the implementation of expandable parameters to convert `arg`s within
 *    MSS descriptors that contain std::vector as a data_storage.
 *  - `branch_selector` encapsulates the work with condition tree within `intermediate` class.
 */

#pragma once

#include <functional>
#include <tuple>
#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/meta.hpp"
#include "../../common/tuple_util.hpp"

#include "condition.hpp"

namespace gridtools {
    namespace _impl {
        namespace condition_tree {

            template < typename ComposeLeafs >
            struct compose_trees_f {

                template < typename >
                struct result;

                template < typename... Args >
                using res_t = typename result< compose_trees_f(Args const &...) >::type;

                // This is needed only to make nvcc happy.
                template < typename Lhs, typename Rhs >
                struct result< compose_trees_f(Lhs &, Rhs &) > : result< compose_trees_f(Lhs const &, Rhs const &) > {};

                template < typename Lhs, typename Rhs >
                struct result< compose_trees_f(Lhs const &, Rhs const &) >
                    : std::result_of< ComposeLeafs(Lhs const &, Rhs const &) > {};

                template < typename LLhs, typename LRhs, typename LC, typename Rhs >
                struct result< compose_trees_f(condition< LLhs, LRhs, LC > const &, Rhs const &) > {
                    using type = condition< res_t< LLhs, Rhs >, res_t< LRhs, Rhs >, LC >;
                };

                template < typename Lhs, typename RLhs, typename RRhs, typename RC >
                struct result< compose_trees_f(Lhs const &, condition< RLhs, RRhs, RC > const &) > {
                    using type = condition< res_t< Lhs, RLhs >, res_t< Lhs, RRhs >, RC >;
                };

                template < typename LLhs, typename LRhs, typename LC, typename RLhs, typename RRhs, typename RC >
                struct result< compose_trees_f(
                    condition< LLhs, LRhs, LC > const &, condition< RLhs, RRhs, RC > const &) > {
                    using rhs_t = condition< RLhs, RRhs, RC >;
                    using type = condition< res_t< LLhs, rhs_t >, res_t< LRhs, rhs_t >, LC >;
                };

                ComposeLeafs m_compose_leafs;

                template < typename Lhs, typename Rhs >
                res_t< Lhs, Rhs > operator()(Lhs const &lhs, Rhs const &rhs) const {
                    return m_compose_leafs(lhs, rhs);
                }

                template < typename LLhs, typename LRhs, typename LC, typename Rhs >
                res_t< condition< LLhs, LRhs, LC >, Rhs > operator()(
                    condition< LLhs, LRhs, LC > const &lhs, Rhs const &rhs) const {
                    return {lhs.m_value, this->operator()(lhs.m_first, rhs), this->operator()(lhs.m_second, rhs)};
                }

                template < typename Lhs, typename RLhs, typename RRhs, typename RC >
                res_t< Lhs, condition< RLhs, RRhs, RC > > operator()(
                    Lhs const &lhs, condition< RLhs, RRhs, RC > const &rhs) const {
                    return {rhs.m_value, this->operator()(lhs, rhs.m_first), this->operator()(lhs, rhs.m_second)};
                }

                template < typename LLhs, typename LRhs, typename LC, typename RLhs, typename RRhs, typename RC >
                res_t< condition< LLhs, LRhs, LC >, condition< RLhs, RRhs, RC > > operator()(
                    condition< LLhs, LRhs, LC > const &lhs, condition< RLhs, RRhs, RC > const &rhs) const {
                    return {lhs.m_value, this->operator()(lhs.m_first, rhs), this->operator()(lhs.m_second, rhs)};
                }
            };

            struct compose_leafs_f {
                template < typename S, typename T >
                auto operator()(S &&seq, T &&elem) const GT_AUTO_RETURN(
                    tuple_util::deep_copy(tuple_util::push_back(std::forward< S >(seq), std::forward< T >(elem))));
            };

            template < typename Trees >
            auto make_tree_from_forest(Trees &&trees) GT_AUTO_RETURN(
                tuple_util::fold(compose_trees_f< compose_leafs_f >{}, std::tuple<>{}, std::forward< Trees >(trees)));

            template < typename TransformLeaf >
            struct transform_f {
                template < typename >
                struct result;

                template < typename T >
                using res_t = typename result< transform_f(T const &) >::type;

                template < typename T >
                struct result< transform_f(T const &) > {
                    using type = typename std::result_of< TransformLeaf(T const &) >::type;
                };

                template < typename Lhs, typename Rhs, typename C >
                struct result< transform_f(condition< Lhs, Rhs, C > const &) > {
                    using type = condition< res_t< Lhs >, res_t< Rhs >, C >;
                };

                TransformLeaf m_transform_leaf;

                template < typename T >
                res_t< T > operator()(T const &src) const {
                    return m_transform_leaf(src);
                }

                template < typename Lhs, typename Rhs, typename C >
                res_t< condition< Lhs, Rhs, C > > operator()(condition< Lhs, Rhs, C > const &src) const {
                    return {src.m_value, this->operator()(src.m_first), this->operator()(src.m_second)};
                }
            };

            template < typename Fun >
            struct apply_with_tree_f {
                template < typename >
                struct result;

                template < typename T >
                using res_t = typename result< apply_with_tree_f(T const &) >::type;

                template < typename T >
                struct result< apply_with_tree_f(T const &) > {
                    using type = typename std::result_of< Fun(T const &) >::type;
                };

                template < typename Lhs, typename Rhs, typename C >
                struct result< apply_with_tree_f(condition< Lhs, Rhs, C > const &) > {
                    using type = typename std::common_type< res_t< Lhs >, res_t< Rhs > >::type;
                };

                Fun m_fun;

                template < typename T >
                res_t< T > operator()(T const &leaf) const {
                    return m_fun(leaf);
                }

                template < typename Lhs, typename Rhs, typename C >
                res_t< condition< Lhs, Rhs, C > > operator()(condition< Lhs, Rhs, C > const &node) const {
                    return node.m_value() ? this->operator()(node.m_first) : this->operator()(node.m_second);
                }
            };

            template < typename Fun >
            apply_with_tree_f< typename std::decay< Fun >::type > apply_with_tree(Fun &&fun) {
                return {fun};
            }

            template < typename T >
            struct lazy_all_leaves_in_tree {
                using type = std::tuple< typename std::decay< T >::type >;
            };
            template < class T >
            using all_leaves_in_tree = typename lazy_all_leaves_in_tree< T >::type;

            template < typename Lhs, typename Rhs, typename Cond >
            struct lazy_all_leaves_in_tree< condition< Lhs, Rhs, Cond > > {
                using type = meta::concat< all_leaves_in_tree< Lhs >, all_leaves_in_tree< Rhs > >;
            };

            template < typename... Trees >
            using all_leaves_in_forest =
                meta::flatten< meta::apply< meta::transform< all_leaves_in_tree >, std::tuple< Trees... > > >;
        }
    }

    /// Check that the object is a condition tree and all leafs satisfy the given predicate.
    template < typename Leaf, template < typename > class Pred >
    struct is_condition_tree_of : Pred< Leaf > {};

    template < typename Lhs, typename Rhs, typename Condition, template < typename > class Pred >
    struct is_condition_tree_of< condition< Lhs, Rhs, Condition >, Pred >
        : meta::conjunction< is_condition_tree_of< Lhs, Pred >, is_condition_tree_of< Rhs, Pred > > {};

    /// Transforms the condition tree by applying to all leaves the given functor
    template < typename Tree, typename Fun >
    auto condition_tree_transform(Tree &&tree, Fun &&fun)
        GT_AUTO_RETURN(_impl::condition_tree::transform_f< typename std::decay< Fun >::type >{std::forward< Fun >(fun)}(
            std::forward< Tree >(tree)));

    /**
     *  A helper for runtime dispatch on a sequence of condition trees.
     *
     *  The class does the following:
     *    - takes a sequence of trees in constructor [types of the trees are template parameters];
     *    - the original sequence trees is composed into the tree of fusion sequences [those sequences are called
     *      branches below]
     *    - there is `apply` method, that performs the evaluation of conditions within the nodes and invokes
     *      the provided functor with the choosen branch as a first argument.
     *
     * @tparam Trees - condition trees
     */
    template < typename... Trees >
    class branch_selector {
        using tree_t = decltype(_impl::condition_tree::make_tree_from_forest(std::declval< std::tuple< Trees... > >()));
        tree_t m_tree;

      public:
        /// An std  tuple containing all leaves of all trees. I.e. the flat view for all trees.
        using all_leaves_t = _impl::condition_tree::all_leaves_in_forest< Trees... >;

        template < class Seq >
        branch_selector(Seq &&seq)
            : m_tree(_impl::condition_tree::make_tree_from_forest(std::forward< Seq >(seq))) {
            static_assert(meta::length< Seq >::value == sizeof...(Trees), "");
        }

        /**
         *  Performs the evaluation of conditions in the trees;
         *  chooses the sequence of leafs (one per each tree) based on that evaluation
         *  applies the provided functor with the chosen sequence as a first parameter.
         *
         *  Example (simplified usage pattern in `intermediate` class):
         *  \verbatim
         *    template <class... MssTrees> class intermediate {
         *      branch_selector<MssTrees...> m_selector;
         *      struct run_f {
         *        template <class MssFlatSequence>
         *        void operator()(MssFlatSequence mss_sequence) const {
         *           // Do the actual computation here.
         *           // At this point all conditions are evaluated and the correspondent
         *           // flat sequence is provided to the caller as a parameter.
         *        }
         *      };
         *    public:
         *      void run {
         *        m_selector.apply(run_f{});
         *      }
         *    };
         *  \endverbatim
         *
         * @tparam Fun - the type of the functor to be invoked.
         * @tparam Args - the types of the rest of the arguments that are passed to the functor after the branch
         * @param fun - the functor to be invoked : `fun(<selected_branch>, args...)`
         * @param args - the rest of the arguments that are passed to the functor after the branch
         * @return - what `fun` invocation actually returns. The result type is calculated as a std::common_type
         *           functor invocations return values for all possible branches.
         */
        template < typename Fun, typename... Args >
        auto apply(Fun &&fun, Args &&... args) const GT_AUTO_RETURN((_impl::condition_tree::apply_with_tree(
            std::bind(std::forward< Fun >(fun), std::placeholders::_1, std::forward< Args >(args)...))(m_tree)));
    };

    /// Empty case specialization.
    template <>
    class branch_selector<> {
      public:
        using all_leaves_t = std::tuple<>;

        branch_selector(std::tuple<> &&) {}

        template < typename Fun, typename... Args >
        auto apply(Fun &&fun, Args &&... args) const
            GT_AUTO_RETURN((std::forward< Fun >(fun)(std::tuple<>{}, std::forward< Args >(args)...)));
    };

    /// Generator for branch_selector
    template < typename... Trees >
    branch_selector< typename std::decay< Trees >::type... > make_branch_selector(Trees &&... trees) {
        return std::make_tuple(std::forward< Trees >(trees)...);
    };
}

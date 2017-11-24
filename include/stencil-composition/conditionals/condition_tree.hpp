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

#include <type_traits>
#include <functional>

#include <boost/fusion/include/fold.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/push_back.hpp>
#include <boost/fusion/include/single_view.hpp>
#include <boost/fusion/include/joint_view.hpp>
#include <boost/fusion/include/is_sequence.hpp>
#include <boost/fusion/include/as_vector.hpp>

#include <boost/mpl/logical.hpp>
#include <boost/mpl/empty.hpp>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/copy_into_variadic.hpp"
#include "../../common/generic_metafunctions/is_sequence_of.hpp"

#include "condition.hpp"
#include "conditional.hpp"

namespace gridtools {

    namespace _impl {
        template < template < typename, template < typename > class > class ContainerPred,
            template < typename > class ElementPred >
        struct predicate_compose {
            template < typename Container >
            using apply = ContainerPred< Container, ElementPred >;
        };

        struct push_back_f {
            template < typename S, typename T >
            auto operator()(S &&sec, T &&elem) const GT_AUTO_RETURN(
                boost::fusion::as_vector(boost::fusion::push_back(std::forward< S >(sec), std::forward< T >(elem))));
        };

        template < typename ComposeLeafs >
        struct compose_condition_trees_f {

            template < typename >
            struct result;

            template < typename... Args >
            using res_t = typename result< compose_condition_trees_f(Args const &...) >::type;

            template < typename Lhs, typename Rhs >
            struct result< compose_condition_trees_f(Lhs const &, Rhs const &) >
                : std::result_of< ComposeLeafs(Lhs const &, Rhs const &) > {};

            template < typename LLhs, typename LRhs, typename LC, typename Rhs >
            struct result< compose_condition_trees_f(condition< LLhs, LRhs, LC > const &, Rhs const &) > {
                using type = condition< res_t< LLhs, Rhs >, res_t< LRhs, Rhs >, LC >;
            };

            template < typename Lhs, typename RLhs, typename RRhs, typename RC >
            struct result< compose_condition_trees_f(Lhs const &, condition< RLhs, RRhs, RC > const &) > {
                using type = condition< res_t< Lhs, RLhs >, res_t< Lhs, RRhs >, RC >;
            };

            template < typename LLhs, typename LRhs, typename LC, typename RLhs, typename RRhs, typename RC >
            struct result< compose_condition_trees_f(
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
                return {lhs.value(), this->operator()(lhs.first(), rhs), this->operator()(lhs.second(), rhs)};
            }

            template < typename Lhs, typename RLhs, typename RRhs, typename RC >
            res_t< Lhs, condition< RLhs, RRhs, RC > > operator()(
                Lhs const &lhs, condition< RLhs, RRhs, RC > const &rhs) const {
                return {rhs.value(), this->operator()(lhs, rhs.first()), this->operator()(lhs, rhs.second())};
            }

            template < typename LLhs, typename LRhs, typename LC, typename RLhs, typename RRhs, typename RC >
            res_t< condition< LLhs, LRhs, LC >, condition< RLhs, RRhs, RC > > operator()(
                condition< LLhs, LRhs, LC > const &lhs, condition< RLhs, RRhs, RC > const &rhs) const {
                return {lhs.value(), this->operator()(lhs.first(), rhs), this->operator()(lhs.second(), rhs)};
            }
        };

        template < typename ComposeLeafs = push_back_f >
        compose_condition_trees_f< typename std::decay< ComposeLeafs >::type > compose_condition_trees(
            ComposeLeafs &&compose_leafs = {}) {
            return {std::forward< ComposeLeafs >(compose_leafs)};
        }

        struct condition_leaves_view_f {
            template < typename >
            struct result;

            template < typename T >
            using res_t = typename result< condition_leaves_view_f(T &) >::type;

            template < typename T >
            struct result< condition_leaves_view_f(T &) > {
                using type = boost::fusion::single_view< T & >;
            };

            template < typename Lhs, typename Rhs, typename C >
            struct result< condition_leaves_view_f(condition< Lhs, Rhs, C > &) > {
                using type = boost::fusion::joint_view< const res_t< Lhs >, const res_t< Rhs > >;
            };

            template < typename T >
            res_t< T > operator()(T &leaf) const {
                return res_t< T >{leaf};
            }

            template < typename Lhs, typename Rhs, typename C >
            res_t< condition< Lhs, Rhs, C > > operator()(condition< Lhs, Rhs, C > &node) const {
                auto a = this->operator()(node.first());
                return {this->operator()(node.first()), this->operator()(node.second())};
            }
        };

        template < typename Fun >
        struct apply_with_condtion_tree_f {
            template < typename >
            struct result;

            template < typename T >
            using res_t = typename result< apply_with_condtion_tree_f(T &) >::type;

            template < typename T >
            struct result< apply_with_condtion_tree_f(T &) > {
                using type = typename std::result_of< Fun(T &) >::type;
            };

            template < typename Lhs, typename Rhs, typename C >
            struct result< apply_with_condtion_tree_f(condition< Lhs, Rhs, C > &) > {
                using type = typename std::common_type< res_t< Lhs >, res_t< Rhs > >::type;
            };

            template < typename Lhs, typename Rhs, typename C >
            struct result< apply_with_condtion_tree_f(condition< Lhs, Rhs, C > const &) > {
                using type = typename std::common_type< res_t< Lhs const >, res_t< Rhs const > >::type;
            };

            Fun m_fun;

            template < typename T >
            res_t< T > operator()(T &leaf) const {
                return m_fun(leaf);
            }

            template < typename Lhs, typename Rhs, typename C >
            res_t< condition< Lhs, Rhs, C > > operator()(condition< Lhs, Rhs, C > &node) const {
                return node.value().value() ? this->operator()(node.first()) : this->operator()(node.second());
            }

            template < typename Lhs, typename Rhs, typename C >
            res_t< condition< Lhs, Rhs, C > const > operator()(condition< Lhs, Rhs, C > const &node) const {
                return node.value().value() ? this->operator()(node.first()) : this->operator()(node.second());
            }
        };

        template < typename Fun, typename... Args >
        apply_with_condtion_tree_f< Fun > apply_with_condtion_tree(Fun &&fun, Args &&... args) {
            return {fun};
        }
    }

    template < typename Leaf, template < typename > class Pred >
    struct is_condition_tree_of : Pred< Leaf > {};

    template < typename Lhs, typename Rhs, uint_t Tag, uint_t SwitchId, template < typename > class Pred >
    struct is_condition_tree_of< condition< Lhs, Rhs, conditional< Tag, SwitchId > >, Pred >
        : boost::mpl::and_< is_condition_tree_of< Lhs, Pred >, is_condition_tree_of< Rhs, Pred > > {};

    template < typename T, template < typename > class Pred >
    using is_condition_forest_of =
        is_sequence_of< T, _impl::predicate_compose< is_condition_tree_of, Pred >::template apply >;

    template < typename T, template < typename > class Pred >
    using is_condition_tree_of_sequence_of =
        is_condition_tree_of< T, _impl::predicate_compose< is_sequence_of, Pred >::template apply >;

    template < typename Lhs, typename Rhs, typename LeavesEqual = std::is_same< boost::mpl::_1, boost::mpl::_2 > >
    struct condition_tree_equal : boost::mpl::apply< LeavesEqual, Lhs, Rhs >::type {};

    template < typename LLhs,
        typename LRhs,
        typename LConditional,
        typename RLhs,
        typename RRhs,
        typename RConditional,
        typename LeavesEqual >
    struct condition_tree_equal< condition< LLhs, LRhs, LConditional >,
        condition< RLhs, RRhs, RConditional >,
        LeavesEqual > : boost::mpl::and_< std::is_same< LConditional, RConditional >,
                            condition_tree_equal< LLhs, RLhs, LeavesEqual >,
                            condition_tree_equal< LRhs, RRhs, LeavesEqual > > {};

    template < typename Forest >
    auto make_condition_tree_from_forest(Forest &&forest) GT_AUTO_RETURN(boost::fusion::fold(
        std::forward< Forest >(forest), boost::fusion::make_vector(), _impl::compose_condition_trees()));

    // TODO(anstaf): add condition_tree_transform.

    template < typename Forest >
    class branch_selector {
        GRIDTOOLS_STATIC_ASSERT(boost::fusion::traits::is_sequence< Forest >::value, "Forest should be a sequence");
        GRIDTOOLS_STATIC_ASSERT(!boost::mpl::empty< Forest >::value, "Forest should not be empty.");

        using tree_t = decltype(make_condition_tree_from_forest(std::declval< Forest >()));
        tree_t m_tree;

      public:
        using const_branches_t = typename std::result_of< _impl::condition_leaves_view_f(tree_t const &) >::type;
        using branches_t = typename std::result_of< _impl::condition_leaves_view_f(tree_t &) >::type;

        branch_selector(Forest const &src) : m_tree(make_condition_tree_from_forest(src)) {}
        branch_selector(Forest &&src) noexcept : m_tree(make_condition_tree_from_forest(std::move(src))) {}

        auto branches() const GT_AUTO_RETURN(_impl::condition_leaves_view_f{}(m_tree));
        auto branches() GT_AUTO_RETURN(_impl::condition_leaves_view_f{}(m_tree));

        // TODO(anstaf): add mutable version and test it
        template < typename Fun, typename... Args >
        auto apply(Fun &&fun, Args &&... args) const GT_AUTO_RETURN(_impl::apply_with_condtion_tree(
            std::bind(std::forward< Fun >(fun), std::placeholders::_1, std::forward< Args >(args)...))(m_tree));
    };

    template < typename T,
        typename Forest = typename std::decay< T >::type,
        typename std::enable_if< boost::fusion::traits::is_sequence< Forest >::value, int >::type = 0 >
    branch_selector< Forest > make_branch_selector(T &&src) {
        return src;
    }

    template < typename T,
        typename Tree = typename std::decay< T >::type,
        typename std::enable_if< !boost::fusion::traits::is_sequence< Tree >::value, int >::type = 0 >
    auto make_branch_selector(T &&tree)
        GT_AUTO_RETURN(make_branch_selector(boost::fusion::make_vector(std::forward< T >(tree))));

    template < typename Tree1, typename Tree2, typename... Trees >
    auto make_branch_selector(Tree1 &&tree1, Tree2 &&tree2, Trees &&... trees)
        GT_AUTO_RETURN(make_branch_selector(boost::fusion::make_vector(
            std::forward< Tree1 >(tree1), std::forward< Tree2 >(tree2), std::forward< Trees >(trees)...)));
}

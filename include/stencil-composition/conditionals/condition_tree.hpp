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
#include <boost/fusion/include/transform.hpp>

#include <boost/mpl/back_inserter.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/logical.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/single_view.hpp>
#include <boost/mpl/joint_view.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/empty_sequence.hpp>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/is_sequence_of.hpp"

#include "condition.hpp"
#include "conditional.hpp"

namespace gridtools {

    namespace _impl {
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

        template < typename... Trees >
        auto make_condition_tree_from_forest(Trees &&... trees)
            GT_AUTO_RETURN(boost::fusion::fold(boost::fusion::make_vector(std::forward< Trees >(trees)...),
                boost::fusion::make_vector(),
                _impl::compose_condition_trees()));

        template < typename TransformLeaf >
        struct condition_transform_f {
            template < typename >
            struct result;

            template < typename T >
            using res_t = typename result< condition_transform_f(T const &) >::type;

            template < typename T >
            struct result< condition_transform_f(T const &) > {
                using type = typename std::result_of< TransformLeaf(T const &) >::type;
            };

            template < typename Lhs, typename Rhs, typename C >
            struct result< condition_transform_f(condition< Lhs, Rhs, C > const &) > {
                using type = condition< res_t< Lhs >, res_t< Rhs >, C >;
            };

            TransformLeaf m_transform_leaf;

            template < typename T >
            res_t< T > operator()(T const &src) const {
                return m_transform_leaf(src);
            }

            template < typename Lhs, typename Rhs, typename C >
            res_t< condition< Lhs, Rhs, C > > operator()(condition< Lhs, Rhs, C > const &src) const {
                return {src.value(), this->operator()(src.first()), this->operator()(src.second())};
            }
        };

        template < typename Fun >
        struct apply_with_condtion_tree_f {
            template < typename >
            struct result;

            template < typename T >
            using res_t = typename result< apply_with_condtion_tree_f(T const &) >::type;

            template < typename T >
            struct result< apply_with_condtion_tree_f(T const &) > {
                using type = typename std::result_of< Fun(T const &) >::type;
            };

            template < typename Lhs, typename Rhs, typename C >
            struct result< apply_with_condtion_tree_f(condition< Lhs, Rhs, C > const &) > {
                using type = typename std::common_type< res_t< Lhs >, res_t< Rhs > >::type;
            };

            Fun m_fun;

            template < typename T >
            res_t< T > operator()(T const &leaf) const {
                return m_fun(leaf);
            }

            template < typename Lhs, typename Rhs, typename C >
            res_t< condition< Lhs, Rhs, C > > operator()(condition< Lhs, Rhs, C > const &node) const {
                return node.value().value() ? this->operator()(node.first()) : this->operator()(node.second());
            }
        };

        template < typename Fun >
        apply_with_condtion_tree_f< Fun > apply_with_condtion_tree(Fun &&fun) {
            return {fun};
        }

        template < typename T >
        struct all_leaves_in_tree {
            using type = boost::mpl::single_view< typename std::decay< T >::type >;
        };

        template < typename Lhs, typename Rhs, typename Cond >
        struct all_leaves_in_tree< condition< Lhs, Rhs, Cond > > {
            using type = boost::mpl::joint_view< typename all_leaves_in_tree< Lhs >::type,
                typename all_leaves_in_tree< Rhs >::type >;
        };

        template < typename... Trees >
        using all_leaves_in_forest_t = typename boost::mpl::fold< boost::mpl::vector< Trees... >,
            boost::mpl::empty_sequence,
            boost::mpl::joint_view< boost::mpl::_1, all_leaves_in_tree< std::decay< boost::mpl::_2 > > > >::type;
    }

    template < typename Leaf, template < typename > class Pred >
    struct is_condition_tree_of : Pred< Leaf > {};

    template < typename Lhs, typename Rhs, uint_t Tag, uint_t SwitchId, template < typename > class Pred >
    struct is_condition_tree_of< condition< Lhs, Rhs, conditional< Tag, SwitchId > >, Pred >
        : boost::mpl::and_< is_condition_tree_of< Lhs, Pred >, is_condition_tree_of< Rhs, Pred > > {};

    template < typename Tree, typename Fun >
    auto condition_tree_transform(Tree &&tree, Fun &&fun)
        GT_AUTO_RETURN(_impl::condition_transform_f< Fun >{std::forward< Fun >(fun)}(std::forward< Tree >(tree)));

    template < typename... Trees >
    class branch_selector {
        using tree_t = decltype(_impl::make_condition_tree_from_forest(std::declval< Trees >()...));
        tree_t m_tree;

      public:
        using all_leaves_t = typename boost::mpl::copy< _impl::all_leaves_in_forest_t< Trees... >,
            boost::mpl::back_inserter< boost::mpl::vector0<> > >::type;

        branch_selector(Trees const &... trees) : m_tree(_impl::make_condition_tree_from_forest(trees...)) {}
        branch_selector(Trees &&... trees) noexcept
            : m_tree(_impl::make_condition_tree_from_forest(std::move(trees)...)) {}

        template < typename Fun, typename... Args >
        auto apply(Fun &&fun, Args &&... args) const GT_AUTO_RETURN(_impl::apply_with_condtion_tree(
            std::bind(std::forward< Fun >(fun), std::placeholders::_1, std::forward< Args >(args)...))(m_tree));
    };
    template < typename Tree, typename... Trees >
    branch_selector< typename std::decay< Tree >::type, typename std::decay< Trees >::type... > make_branch_selector(
        Tree &&tree, Trees &&... trees) {
        return {std::forward< Tree >(tree), std::forward< Trees >(trees)...};
    };
}

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
#include <stencil-composition/conditionals/condition_tree.hpp>

#include <array>
#include <functional>

#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/include/comparison.hpp>
#include <boost/fusion/include/transform.hpp>

#include <gtest/gtest.h>

#include <common/functional.hpp>

#include <stencil-composition/conditionals/condition.hpp>
#include <stencil-composition/conditionals/conditional.hpp>

namespace gridtools {
    namespace {
        namespace f = boost::fusion;

        using conditional_t = conditional< 42 >;

        template < typename Lhs, typename Rhs >
        using node_t = condition< Lhs, Rhs, conditional< 42 > >;

        template < typename... Ts >
        using seq_t = f::vector< Ts... >;

        static_assert(is_condition_tree_of< int *, std::is_pointer >{}, "good leaf");
        static_assert(is_condition_tree_of< node_t< int *, void * >, std::is_pointer >{}, "good node");
        static_assert(
            is_condition_tree_of< node_t< int *, node_t< long *, void * > >, std::is_pointer >{}, "good tree");
        static_assert(!is_condition_tree_of< int, std::is_pointer >{}, "bad leaf");
        static_assert(!is_condition_tree_of< node_t< int *, int >, std::is_pointer >{}, "bad node");
        static_assert(!is_condition_tree_of< node_t< int *, node_t< long, void * > >, std::is_pointer >{}, "bad tree");

        static_assert(is_condition_tree_of_sequence_of< seq_t<>, std::is_pointer >::value, "empty");
        static_assert(is_condition_tree_of_sequence_of< seq_t< int * >, std::is_pointer >::value, "good leaf");
        static_assert(
            is_condition_tree_of_sequence_of< node_t< seq_t< int * >, seq_t< void * > >, std::is_pointer >::value,
            "good node");

        static_assert(is_condition_forest_of< seq_t<>, std::is_pointer >::value, "empty");
        static_assert(is_condition_forest_of< seq_t< int * >, std::is_pointer >::value, "good leaf seq");
        static_assert(
            is_condition_forest_of< seq_t< node_t< int *, void * > >, std::is_pointer >::value, "good node seq");

        static_assert(condition_tree_equal< int, int >{}, "same leaves");
        static_assert(condition_tree_equal< node_t< int, double >, node_t< int, double > >{}, "same nodes");
        static_assert(!condition_tree_equal< node_t< int, double >, int >{}, "node vs leaf");
        static_assert(!condition_tree_equal< int, void * >{}, "different leaves");
        static_assert(!condition_tree_equal< node_t< int, double >, node_t< int, int > >{}, "different nodes");
        static_assert(!condition_tree_equal< node_t< int, double >, condition< int, double, conditional< 0 > > >{},
            "different tags");

        template < typename Lhs, typename Rhs >
        node_t< Lhs, Rhs > make_node(Lhs lhs, Rhs rhs, std::function< bool() > fun = {}) {
            return make_condition(conditional_t{fun}, lhs, rhs);
        };

        TEST(branch_selector, minimalistic) {
            auto testee = make_branch_selector(1);
            EXPECT_EQ(testee.branches(), f::make_vector(f::make_vector(1)));
            EXPECT_EQ(testee.apply(identity{}), f::make_vector(1));
        }

        TEST(branch_selector, mutable_branches) {
            auto testee = make_branch_selector(1);
            f::at_c< 0 >(f::at_c< 0 >(testee.branches())) = 2;
            EXPECT_EQ(testee.branches(), f::make_vector(f::make_vector(2)));
        }

        TEST(branch_selector, no_conditions) {
            auto testee = make_branch_selector(1, 2);
            EXPECT_EQ(testee.branches(), f::make_vector(f::make_vector(1, 2)));
            EXPECT_EQ(testee.apply(identity{}), f::make_vector(1, 2));
        }

        TEST(branch_selector, one_condition) {
            bool key;
            auto testee = make_branch_selector(make_node(1, 2, [&] { return key; }));
            EXPECT_EQ(testee.branches(), f::make_vector(f::make_vector(1), f::make_vector(2)));
            key = true;
            EXPECT_EQ(testee.apply(identity{}), f::make_vector(1));
            key = false;
            EXPECT_EQ(testee.apply(identity{}), f::make_vector(2));
        }

        TEST(branch_selector, one_condition_with_prefix) {
            bool key;
            auto testee = make_branch_selector(0, make_node(1, 2, [&] { return key; }));
            EXPECT_EQ(testee.branches(), f::make_vector(f::make_vector(0, 1), f::make_vector(0, 2)));
            key = true;
            EXPECT_EQ(testee.apply(identity{}), f::make_vector(0, 1));
            key = false;
            EXPECT_EQ(testee.apply(identity{}), f::make_vector(0, 2));
        }

        TEST(branch_selector, one_condition_with_suffix) {
            bool key;
            auto testee = make_branch_selector(make_node(1, 2, [&] { return key; }), 3);
            EXPECT_EQ(testee.branches(), f::make_vector(f::make_vector(1, 3), f::make_vector(2, 3)));
            key = true;
            EXPECT_EQ(testee.apply(identity{}), f::make_vector(1, 3));
            key = false;
            EXPECT_EQ(testee.apply(identity{}), f::make_vector(2, 3));
        }

        TEST(branch_selector, condition_tree) {
            std::array< bool, 2 > keys;
            auto testee =
                make_branch_selector(make_node(1, make_node(2, 3, [&] { return keys[1]; }), [&] { return keys[0]; }));
            EXPECT_EQ(testee.branches(), f::make_vector(f::make_vector(1), f::make_vector(2), f::make_vector(3)));
            keys = {true};
            EXPECT_EQ(testee.apply(identity{}), f::make_vector(1));
            keys = {false, true};
            EXPECT_EQ(testee.apply(identity{}), f::make_vector(2));
            keys = {false, false};
            EXPECT_EQ(testee.apply(identity{}), f::make_vector(3));
        }

        TEST(branch_selector, two_conditions) {
            std::array< bool, 2 > keys;
            auto testee = make_branch_selector(
                make_node(1, 2, [&] { return keys[0]; }), make_node(10, 20, [&] { return keys[1]; }));
            EXPECT_EQ(testee.branches(),
                f::make_vector(
                          f::make_vector(1, 10), f::make_vector(1, 20), f::make_vector(2, 10), f::make_vector(2, 20)));
            keys = {true, true};
            EXPECT_EQ(testee.apply(identity{}), f::make_vector(1, 10));
            keys = {true, false};
            EXPECT_EQ(testee.apply(identity{}), f::make_vector(1, 20));
            keys = {false, true};
            EXPECT_EQ(testee.apply(identity{}), f::make_vector(2, 10));
            keys = {false, false};
            EXPECT_EQ(testee.apply(identity{}), f::make_vector(2, 20));
        }

        template < size_t I >
        using val_t = std::integral_constant< size_t, I >;

        struct get_elem_f {
            template < typename T, T I >
            T operator()(std::integral_constant< T, I >) const {
                return I;
            }
        };

        template < typename Fun >
        struct transform_f {
            Fun m_fun;
            template < typename Sec >
            auto operator()(const Sec &sec) const GT_AUTO_RETURN(f::as_vector(f::transform(sec, m_fun)));
        };

        TEST(branch_selector, different_types) {
            bool key;
            auto testee = make_branch_selector(make_node(val_t< 1 >{}, val_t< 2 >{}, [&] { return key; }));
            EXPECT_EQ(testee.branches(), f::make_vector(f::make_vector(val_t< 1 >{}), f::make_vector(val_t< 2 >{})));
            key = true;
            transform_f< get_elem_f > fun;
            EXPECT_EQ(testee.apply(fun), f::make_vector(1));
            key = false;
            EXPECT_EQ(testee.apply(fun), f::make_vector(2));
        }

        struct second_f {
            template < typename T, typename U >
            U operator()(T t, U u) const {
                return u;
            }
        };

        TEST(branch_selector, extra_args) { EXPECT_EQ(make_branch_selector(1).apply(second_f{}, 8), 8); }

        TEST(branch_selector, void_return) { make_branch_selector(1).apply(noop{}); }
    }
}

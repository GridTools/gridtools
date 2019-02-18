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
#include <gridtools/stencil-composition/conditionals/condition_tree.hpp>

#include <array>
#include <functional>
#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/functional.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/stencil-composition/conditionals/condition.hpp>

namespace gridtools {
    namespace {
        template <typename Lhs, typename Rhs>
        using node_t = condition<Lhs, Rhs, std::function<bool()>>;

        static_assert(is_condition_tree_of<int *, std::is_pointer>::value, "good leaf");
        static_assert(is_condition_tree_of<node_t<int *, void *>, std::is_pointer>::value, "good node");
        static_assert(is_condition_tree_of<node_t<int *, node_t<long *, void *>>, std::is_pointer>::value, "good tree");
        static_assert(!is_condition_tree_of<int, std::is_pointer>::value, "bad leaf");
        static_assert(!is_condition_tree_of<node_t<int *, int>, std::is_pointer>::value, "bad node");
        static_assert(!is_condition_tree_of<node_t<int *, node_t<long, void *>>, std::is_pointer>::value, "bad tree");

        template <typename Lhs, typename Rhs, typename Cond = std::function<bool()>>
        condition<Lhs, Rhs, Cond> make_node(Lhs lhs, Rhs rhs, Cond fun = {}) {
            return {fun, lhs, rhs};
        };

        TEST(tree_transform, leaf) { EXPECT_EQ(condition_tree_transform(1, std::negate<int>{}), -1); }

        TEST(tree_transform, node) {
            auto actual = condition_tree_transform(make_node(1, 2), std::negate<int>{});
            EXPECT_EQ(actual.m_first, -1);
            EXPECT_EQ(actual.m_second, -2);
        }

        struct checker_f {
            template <class Actual, class Expected>
            void operator()(Actual const &actual, Expected const &expected) const {
                EXPECT_EQ(actual, expected);
            }
        };

        TEST(branch_selector, empty) {
            auto testee = make_branch_selector();
            static_assert(std::is_same<decltype(testee)::all_leaves_t, std::tuple<>>::value, "all_leaves");
            testee.apply(checker_f{}, std::make_tuple());
        }

        TEST(branch_selector, minimalistic) {
            auto testee = make_branch_selector(1);
            static_assert(std::is_same<decltype(testee)::all_leaves_t, std::tuple<int>>::value, "all_leaves");
            testee.apply(checker_f{}, std::make_tuple(1));
        }

        TEST(branch_selector, no_conditions) {
            auto testee = make_branch_selector(1, 2);
            static_assert(std::is_same<decltype(testee)::all_leaves_t, std::tuple<int, int>>::value, "all_leaves");
            testee.apply(checker_f{}, std::make_tuple(1, 2));
        }

        TEST(branch_selector, one_condition) {
            bool key;
            auto testee = make_branch_selector(make_node(1, 2, [&] { return key; }));
            key = true;
            testee.apply(checker_f{}, std::make_tuple(1));
            key = false;
            testee.apply(checker_f{}, std::make_tuple(2));
        }

        TEST(branch_selector, one_condition_with_prefix) {
            bool key;
            auto testee = make_branch_selector(0, make_node(1, 2, [&] { return key; }));
            key = true;
            testee.apply(checker_f{}, std::make_tuple(0, 1));
            key = false;
            testee.apply(checker_f{}, std::make_tuple(0, 2));
        }

        TEST(branch_selector, one_condition_with_suffix) {
            bool key;
            auto testee = make_branch_selector(make_node(1, 2, [&] { return key; }), 3);
            key = true;
            testee.apply(checker_f{}, std::make_tuple(1, 3));
            key = false;
            testee.apply(checker_f{}, std::make_tuple(2, 3));
        }

        TEST(branch_selector, condition_tree) {
            std::array<bool, 2> keys;
            auto testee =
                make_branch_selector(make_node(1, make_node(2, 3, [&] { return keys[1]; }), [&] { return keys[0]; }));
            static_assert(std::is_same<decltype(testee)::all_leaves_t, std::tuple<int, int, int>>::value, "all_leaves");
            keys = {true};
            testee.apply(checker_f{}, std::make_tuple(1));
            keys = {false, true};
            testee.apply(checker_f{}, std::make_tuple(2));
            keys = {false, false};
            testee.apply(checker_f{}, std::make_tuple(3));
        }

        TEST(branch_selector, two_conditions) {
            std::array<bool, 2> keys;
            auto testee = make_branch_selector(
                make_node(1, 2, [&] { return keys[0]; }), make_node(10, 20, [&] { return keys[1]; }));
            keys = {true, true};
            testee.apply(checker_f{}, std::make_tuple(1, 10));
            keys = {true, false};
            testee.apply(checker_f{}, std::make_tuple(1, 20));
            keys = {false, true};
            testee.apply(checker_f{}, std::make_tuple(2, 10));
            keys = {false, false};
            testee.apply(checker_f{}, std::make_tuple(2, 20));
        }

        TEST(branch_selector, different_types) {
            using namespace literals;
            bool key;
            auto testee = make_branch_selector(make_node(1_c, 2_c, [&] { return key; }));
            key = true;
            testee.apply(checker_f{}, std::make_tuple(1));
            key = false;
            testee.apply(checker_f{}, std::make_tuple(2));
        }
    } // namespace
} // namespace gridtools

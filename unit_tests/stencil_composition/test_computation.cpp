/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/computation.hpp>

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/arg.hpp>
#include <gridtools/tools/backend_select.hpp>

namespace gridtools {
    namespace {

        using storage_traits_t = backend_t::storage_traits_t;

        using data_store_t = storage_traits_t::data_store_t<float_type, storage_traits_t::storage_info_t<0, 1>>;

        using a = arg<0, data_store_t>;
        using b = arg<1, data_store_t>;

        data_store_t data(std::string const &name = "") { return name; }

        struct my_computation {
            size_t m_count = 0;

            template <class... Args, class... DataStores>
            void run(arg_storage_pair<Args, DataStores> const &... args) {
                ++m_count;
            }

            void reset_meter() { m_count = 0; }
            std::string print_meter() const {
                std::ostringstream strm;
                strm << m_count;
                return strm.str();
            }
            size_t get_count() const { return m_count; }
            double get_time() const { return 0.; /* unused */ }

            template <typename Arg>
            static constexpr rt_extent get_arg_extent(Arg) {
                return {0, 0, 0, 0, 0, 0};
            }
            template <typename Arg>
            static constexpr std::integral_constant<intent, intent::in> get_arg_intent(Arg) {
                return {};
            }
        };

        TEST(computation, default_ctor) {
            EXPECT_FALSE(computation<>{});
            EXPECT_FALSE((computation<a, b>{}));
        }

        TEST(computation, without_args) {
            computation<> testee = my_computation{};
            EXPECT_EQ(testee.print_meter(), "0");
            EXPECT_EQ(testee.get_count(), 0);
            testee.run();
            testee.run();
            EXPECT_EQ(testee.print_meter(), "2");
            EXPECT_EQ(testee.get_count(), 2);
            EXPECT_EQ(testee.print_meter(), "2");
            testee.reset_meter();
            EXPECT_EQ(testee.get_count(), 0);
            EXPECT_EQ(testee.print_meter(), "0");
            // expect compilation failures:
            // testee.run(1);
            // testee.run(a{} = data());
        }

        TEST(computation, move) {
            auto make = []() { return computation<>(my_computation{}); };
            auto testee = make();
            EXPECT_EQ(testee.print_meter(), "0");
        }

        TEST(computation, move_assign) {
            auto make = []() {
                computation<> res = my_computation{};
                res.run();
                return res;
            };
            computation<> testee;
            testee = make();
            EXPECT_EQ(testee.get_count(), 1);
        }

        TEST(computation, one_arg) {
            computation<a> testee = my_computation{};
            testee.run(a{} = data("foo"));
            // expect compilation failures
            // testee.run();
            // testee.run(1);
            // testee.run(b{} = data());
        }

        TEST(computation, many_args) {
            computation<a, b> testee = my_computation{};
            testee.run(a{} = data("foo"), b{} = data("bar"));
            testee.run(b{} = data("bar"), a{} = data("foo"));
        }

        TEST(computation, convertible_args) {
            computation<a, b> tmp = my_computation{};
            tmp.run(a{} = data(), b{} = data());
            computation<b, a> testee = std::move(tmp);
            testee.run(a{} = data(), b{} = data());
            EXPECT_EQ(testee.get_count(), 2);
        }
    } // namespace
} // namespace gridtools

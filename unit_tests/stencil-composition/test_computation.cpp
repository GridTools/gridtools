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

#include <stencil-composition/computation.hpp>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <boost/any.hpp>
#include <stencil-composition/arg.hpp>
#include "backend_select.hpp"

namespace gridtools {
    namespace {

        using storage_traits_t = backend_t::storage_traits_t;

        using data_store_t = storage_traits_t::data_store_t< float_type, storage_traits_t::storage_info_t< 0, 1 > >;

        using a = arg< 0, data_store_t >;
        using b = arg< 1, data_store_t >;

        data_store_t data(std::string const &name = "") { return name; }

        using result = std::vector< std::string >;

        struct my_computation {
            size_t m_count = 0;
            bool m_synced = true;

            template < class... Args, class... DataStores >
            result run(arg_storage_pair< Args, DataStores > const &... args) {
                ++m_count;
                m_synced = false;
                return {args.m_value.name()...};
            }

            void sync_bound_data_stores() { m_synced = true; }
            void reset_meter() { m_count = 0; }
            std::string print_meter() const {
                std::ostringstream strm;
                strm << (m_synced ? "synced" : "not synced") << ":" << m_count;
                return strm.str();
            }
            size_t get_meter() const { return m_count; }
        };

        TEST(computation, default_ctor) {
            EXPECT_FALSE(computation< int >{});
            EXPECT_FALSE((computation< int, a, b >{}));
        }

        TEST(computation, without_args) {
            computation< result > testee = my_computation{};
            static_assert(std::is_same< decltype(testee.run()), result >{}, "");
            EXPECT_EQ(testee.print_meter(), "synced:0");
            EXPECT_EQ(testee.get_meter(), 0);
            testee.run();
            testee.run();
            EXPECT_EQ(testee.print_meter(), "not synced:2");
            EXPECT_EQ(testee.get_meter(), 2);
            testee.sync_bound_data_stores();
            EXPECT_EQ(testee.print_meter(), "synced:2");
            testee.reset_meter();
            EXPECT_EQ(testee.get_meter(), 0);
            EXPECT_EQ(testee.print_meter(), "synced:0");
            // expect compilation failures:
            // testee.run(1);
            // testee.run(a{} = data());
        }

        TEST(computation, void_return) {
            computation< void > testee = my_computation{};
            static_assert(std::is_void< decltype(testee.run()) >{}, "");
            testee.run();
        }

        TEST(computation, move) {
            auto make = []() { return computation< void >(my_computation{}); };
            auto testee = make();
            EXPECT_EQ(testee.print_meter(), "synced:0");
        }

        TEST(computation, move_assign) {
            auto make = []() {
                computation< void > res = my_computation{};
                res.run();
                return res;
            };
            computation< void > testee;
            testee = make();
            EXPECT_EQ(testee.get_meter(), 1);
        }

        TEST(computation, one_arg) {
            computation< result, a > testee = my_computation{};
            EXPECT_EQ(testee.run(a{} = data("foo"))[0], "foo");
            // expect compilation failures
            // testee.run();
            // testee.run(1);
            // testee.run(b{} = data());
        }

        TEST(computation, many_args) {
            computation< result, a, b > testee = my_computation{};
            result res = testee.run(a{} = data("foo"), b{} = data("bar"));
            ASSERT_EQ(res.size(), 2);
            EXPECT_EQ(res[0], "foo");
            EXPECT_EQ(res[1], "bar");
            res = testee.run(b{} = data("bar"), a{} = data("foo"));
            ASSERT_EQ(res.size(), 2);
            EXPECT_EQ(res[0], "foo");
            EXPECT_EQ(res[1], "bar");
        }

        TEST(computation, convertible_args) {
            computation< void, a, b > tmp = my_computation{};
            tmp.run(a{} = data(), b{} = data());
            computation< void, b, a > testee = std::move(tmp);
            testee.run(a{} = data(), b{} = data());
            EXPECT_EQ(testee.get_meter(), 2);
        }

        TEST(computation, convertible_returns) {
            computation< result > tmp = my_computation{};
            tmp.run();
            computation< boost::any > tmp_testee = std::move(tmp);
            tmp_testee.run();
            computation< void > testee = std::move(tmp_testee);
            testee.run();
            EXPECT_EQ(testee.get_meter(), 3);
        }
    }
}

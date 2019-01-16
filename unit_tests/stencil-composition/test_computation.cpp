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

#include <gridtools/stencil-composition/computation.hpp>

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/arg.hpp>
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
            bool m_synced = true;

            template <class... Args, class... DataStores>
            void run(arg_storage_pair<Args, DataStores> const &... args) {
                ++m_count;
                m_synced = false;
            }

            void sync_bound_data_stores() { m_synced = true; }
            void reset_meter() { m_count = 0; }
            std::string print_meter() const {
                std::ostringstream strm;
                strm << (m_synced ? "synced" : "not synced") << ":" << m_count;
                return strm.str();
            }
            size_t get_count() const { return m_count; }
            double get_time() const { return 0.; /* unused */ }

            template <typename Arg>
            rt_extent get_extent(Arg) {
                return {0, 0, 0, 0, 0, 0};
            }
        };

        TEST(computation, default_ctor) {
            EXPECT_FALSE(computation<>{});
            EXPECT_FALSE((computation<a, b>{}));
        }

        TEST(computation, without_args) {
            computation<> testee = my_computation{};
            EXPECT_EQ(testee.print_meter(), "synced:0");
            EXPECT_EQ(testee.get_count(), 0);
            testee.run();
            testee.run();
            EXPECT_EQ(testee.print_meter(), "not synced:2");
            EXPECT_EQ(testee.get_count(), 2);
            testee.sync_bound_data_stores();
            EXPECT_EQ(testee.print_meter(), "synced:2");
            testee.reset_meter();
            EXPECT_EQ(testee.get_count(), 0);
            EXPECT_EQ(testee.print_meter(), "synced:0");
            // expect compilation failures:
            // testee.run(1);
            // testee.run(a{} = data());
        }

        TEST(computation, move) {
            auto make = []() { return computation<>(my_computation{}); };
            auto testee = make();
            EXPECT_EQ(testee.print_meter(), "synced:0");
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

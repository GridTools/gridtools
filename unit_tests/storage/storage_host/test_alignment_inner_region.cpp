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

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil-composition/storage_info_extender.hpp>
#include <gridtools/storage/storage-facility.hpp>
#include <gtest/gtest.h>

namespace gt = gridtools;

template <typename Layout, gt::int_t I>
constexpr gt::uint_t add_or_not(gt::uint_t x) {
    return (Layout::find(1) == I) ? x : 0;
}

template <typename ValueType, gt::uint_t a, typename Layout>
void run() {
    constexpr gt::uint_t h1 = 3;
    constexpr gt::uint_t h2 = 4;
    constexpr gt::uint_t h3 = 5;
    using info = gt::host_storage_info<0, Layout, gt::halo<h1, h2, h3>, gt::alignment<a>>;
    using store = gt::storage_traits<gridtools::target::x86>::data_store_t<ValueType, info>;

    info i(12, 34, 12);
    store s(i);

    auto view = gt::make_host_view(s);

    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(&view(h1, h2, h3)) % a, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(
                  &view(h1 + add_or_not<Layout, 0>(1), h2 + add_or_not<Layout, 1>(1), h3 + add_or_not<Layout, 2>(1))) %
                  a,
        0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(
                  &view(h1 + add_or_not<Layout, 0>(2), h2 + add_or_not<Layout, 1>(2), h3 + add_or_not<Layout, 2>(2))) %
                  a,
        0);
}

template <typename ValueType, gt::uint_t a, typename Layout>
void run_multi() {
    constexpr gt::uint_t h1 = 3;
    constexpr gt::uint_t h2 = 4;
    constexpr gt::uint_t h3 = 5;
    using info = gt::host_storage_info<0, Layout, gt::halo<h1, h2, h3>, gt::alignment<a>>;
    using store = gt::storage_traits<gridtools::target::x86>::data_store_t<ValueType, info>;

    info i(10, 10, 12);
    store s(i, (ValueType)1, "name");
    store t(i, [](int i, int j, int k) { return ValueType(1); }, "name");

    auto views = gt::make_host_view(s);

    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(&views(h1, h2, h3)) % a, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(
                  &views(h1 + add_or_not<Layout, 0>(1), h2 + add_or_not<Layout, 1>(1), h3 + add_or_not<Layout, 2>(1))) %
                  a,
        0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(
                  &views(h1 + add_or_not<Layout, 0>(2), h2 + add_or_not<Layout, 1>(2), h3 + add_or_not<Layout, 2>(2))) %
                  a,
        0);

    auto viewt = gt::make_host_view(t);

    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(&viewt(h1, h2, h3)) % a, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(
                  &viewt(h1 + add_or_not<Layout, 0>(1), h2 + add_or_not<Layout, 1>(1), h3 + add_or_not<Layout, 2>(1))) %
                  a,
        0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(
                  &viewt(h1 + add_or_not<Layout, 0>(2), h2 + add_or_not<Layout, 1>(2), h3 + add_or_not<Layout, 2>(2))) %
                  a,
        0);
}

TEST(Storage, InnerRegionAlignmentChar210) { run<char, 1024, gt::layout_map<2, 1, 0>>(); }

TEST(Storage, InnerRegionAlignmentInt210) { run<int, 256, gt::layout_map<2, 1, 0>>(); }

TEST(Storage, InnerRegionAlignmentFloat210) { run<float, 32, gt::layout_map<2, 1, 0>>(); }

TEST(Storage, InnerRegionAlignmentDouble210) { run<double, 512, gt::layout_map<2, 1, 0>>(); }

TEST(Storage, InnerRegionAlignmentChar012) { run<char, 1024, gt::layout_map<0, 1, 2>>(); }

TEST(Storage, InnerRegionAlignmentInt012) { run<int, 256, gt::layout_map<0, 1, 2>>(); }

TEST(Storage, InnerRegionAlignmentFloat012) { run<float, 32, gt::layout_map<0, 1, 2>>(); }

TEST(Storage, InnerRegionAlignmentDouble012) { run<double, 512, gt::layout_map<0, 1, 2>>(); }

TEST(Storage, InnerRegionAlignmentChar021) { run<char, 1024, gt::layout_map<0, 2, 1>>(); }

TEST(Storage, InnerRegionAlignmentInt021) { run<int, 256, gt::layout_map<0, 2, 1>>(); }

TEST(Storage, InnerRegionAlignmentFloat021) { run<float, 32, gt::layout_map<0, 2, 1>>(); }

TEST(Storage, InnerRegionAlignmentDouble021) { run<double, 512, gt::layout_map<0, 2, 1>>(); }

//////////
TEST(Storage, MultiInnerRegionAlignmentChar210) { run_multi<char, 1024, gt::layout_map<2, 1, 0>>(); }

TEST(Storage, MultiInnerRegionAlignmentInt210) { run_multi<int, 256, gt::layout_map<2, 1, 0>>(); }

TEST(Storage, MultiInnerRegionAlignmentFloat210) { run_multi<float, 32, gt::layout_map<2, 1, 0>>(); }

TEST(Storage, MultiInnerRegionAlignmentDouble210) { run_multi<double, 512, gt::layout_map<2, 1, 0>>(); }

TEST(Storage, MultiInnerRegionAlignmentChar012) { run_multi<char, 1024, gt::layout_map<0, 1, 2>>(); }

TEST(Storage, MultiInnerRegionAlignmentInt012) { run_multi<int, 256, gt::layout_map<0, 1, 2>>(); }

TEST(Storage, MultiInnerRegionAlignmentFloat012) { run_multi<float, 32, gt::layout_map<0, 1, 2>>(); }

TEST(Storage, MultiInnerRegionAlignmentDouble012) { run_multi<double, 512, gt::layout_map<0, 1, 2>>(); }

TEST(Storage, MultiInnerRegionAlignmentChar021) { run_multi<char, 1024, gt::layout_map<0, 2, 1>>(); }

TEST(Storage, MultiInnerRegionAlignmentInt021) { run_multi<int, 256, gt::layout_map<0, 2, 1>>(); }

TEST(Storage, MultiInnerRegionAlignmentFloat021) { run_multi<float, 32, gt::layout_map<0, 2, 1>>(); }

TEST(Storage, MultiInnerRegionAlignmentDouble021) { run_multi<double, 512, gt::layout_map<0, 2, 1>>(); }

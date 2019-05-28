/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "generic_benchmark.hpp"

#include <gridtools/boundary_conditions/boundary.hpp>
#include <gridtools/tools/backend_select.hpp>
#include <gridtools/tools/regression_fixture.hpp>

#include <gtest/gtest.h>

using namespace gridtools;

namespace {
    struct direction_bc_input {
        float_type value;

        GT_FUNCTION
        direction_bc_input() : value(1) {}

        GT_FUNCTION
        direction_bc_input(float_type v) : value(v) {}

        // relative coordinates
        template <typename Direction, typename DataField0, typename DataField1>
        GT_FUNCTION void operator()(
            Direction, DataField0 &data_field0, DataField1 const &data_field1, uint_t i, uint_t j, uint_t k) const {
            data_field1(i, j, k) = data_field0(i, j, k) * value;
        }

        // relative coordinates
        template <sign I, sign K, typename DataField0, typename DataField1>
        GT_FUNCTION void operator()(
            direction<I, minus_, K>, DataField0 &, DataField1 const &data_field1, uint_t i, uint_t j, uint_t k) const {
            data_field1(i, j, k) = 88 * value;
        }

        // relative coordinates
        template <sign K, typename DataField0, typename DataField1>
        GT_FUNCTION void operator()(direction<minus_, minus_, K>,
            DataField0 &,
            DataField1 const &data_field1,
            uint_t i,
            uint_t j,
            uint_t k) const {
            data_field1(i, j, k) = 77777 * value;
        }

        template <typename DataField0, typename DataField1>
        GT_FUNCTION void operator()(direction<minus_, minus_, minus_>,
            DataField0 &,
            DataField1 const &data_field1,
            uint_t i,
            uint_t j,
            uint_t k) const {
            data_field1(i, j, k) = 55555 * value;
        }
    };

    template <typename Src, typename Dst>
    void apply_boundary(array<halo_descriptor, 3> const &halos, Src &src, Dst &dst) {
        gridtools::make_boundary<backend_t>(halos, direction_bc_input{2.0}).apply(src, dst);
    }
    template <typename Src, typename Dst>
    void verify_result(array<halo_descriptor, 3> const &halos, Src &src, Dst &dst) {
        src.sync();
        dst.sync();

        auto src_v = gridtools::make_host_view<access_mode::read_only>(src);
        // auto dst_v = gridtools::make_host_view<access_mode::read_only>(dst);
        auto dst_v = gridtools::make_host_view(dst);

        // check inner domain (should be zero)
        for (uint_t i = halos[0].begin(); i <= halos[0].end(); ++i)
            for (uint_t j = halos[1].begin(); j <= halos[1].end(); ++j)
                for (uint_t k = halos[2].begin(); k <= halos[2].end(); ++k) {
                    EXPECT_EQ(src_v(i, j, k), i + j + k);
                    EXPECT_EQ(dst_v(i, j, k), 0);
                    dst_v(i, j, k) = -1;
                }

        // check corner (direction<minus_, minus_, minus_>)
        for (uint_t i = 0; i < halos[0].begin(); ++i)
            for (uint_t j = 0; j < halos[1].begin(); ++j)
                for (uint_t k = 0; k < halos[2].begin(); ++k) {
                    EXPECT_EQ(dst_v(i, j, k), 111110);
                    dst_v(i, j, k) = -1;
                }

        // check edge (direction<minus_, minus_, K>)
        for (uint_t i = 0; i < halos[0].begin(); ++i)
            for (uint_t j = 0; j < halos[1].begin(); ++j)
                for (uint_t k = halos[2].begin(); k < halos[2].total_length(); ++k) {
                    EXPECT_EQ(dst_v(i, j, k), 155554);
                    dst_v(i, j, k) = -1;
                }

        // check face (direction<I, minus_, K>)
        for (uint_t i = halos[0].begin(); i < halos[0].total_length(); ++i)
            for (uint_t j = 0; j < halos[1].begin(); ++j)
                for (uint_t k = 0; k < halos[2].total_length(); ++k) {
                    EXPECT_EQ(dst_v(i, j, k), 176);
                    dst_v(i, j, k) = -1;
                }

        // remainder
        for (uint_t i = 0; i < halos[0].total_length(); ++i)
            for (uint_t j = halos[1].begin(); j < halos[1].total_length(); ++j)
                for (uint_t k = 0; k < halos[2].total_length(); ++k)
                    if (i < halos[0].begin() || i > halos[0].end() || k < halos[2].begin() || k > halos[2].end() ||
                        j > halos[1].end()) {
                        EXPECT_EQ(dst_v(i, j, k), src_v(i, j, k) * 2);
                        dst_v(i, j, k) = -1;
                    }

        // test the test (all values should be set to -1 now)
        for (uint_t i = 0; i < halos[0].total_length(); ++i)
            for (uint_t j = 0; j < halos[1].total_length(); ++j)
                for (uint_t k = 0; k < halos[2].total_length(); ++k)
                    ASSERT_EQ(dst_v(i, j, k), -1) << i << ", " << j << ", " << k;

        src.sync();
        dst.sync();
    } // namespace
} // namespace

struct distributed_boundary : regression_fixture<3> {};

TEST_F(distributed_boundary, test) {
    auto src = make_storage([](int i, int j, int k) { return i + j + k; });
    auto dst = make_storage(0.f);

    halo_descriptor di{halo_size, halo_size, halo_size, d1() - halo_size - 1, (unsigned)src.info().padded_length<0>()};
    halo_descriptor dj{halo_size, halo_size, halo_size, d2() - halo_size - 1, (unsigned)src.info().padded_length<1>()};
    halo_descriptor dk{halo_size, halo_size, halo_size, d3() - halo_size - 1, (unsigned)src.info().padded_length<2>()};
    array<halo_descriptor, 3> halos{di, dj, dk};

    apply_boundary(halos, src, dst);
    verify_result(halos, src, dst);

    benchmark(generic_benchmark<backend_t>{[&]() { apply_boundary(halos, src, dst); }});
}

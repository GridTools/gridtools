/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/boundary_conditions/apply_gpu.hpp>
#include <gtest/gtest.h>

namespace gt = gridtools;

TEST(apply_gpu, shape) {
    using shape = gt::_impl::kernel_configuration::shape_type;

    {
        shape x(3, 6, 7, 0, 0, 0);
        EXPECT_EQ(x.min(), 3);
        EXPECT_EQ(x.max(), 7);
        EXPECT_EQ(x.median(), 6);
    }
    {
        shape x(3, 3, 3, 0, 0, 0);
        EXPECT_EQ(x.min(), 3);
        EXPECT_EQ(x.max(), 3);
        EXPECT_EQ(x.median(), 3);
    }
    {
        shape x(7, 3, 4, 0, 0, 0);
        EXPECT_EQ(x.min(), 3);
        EXPECT_EQ(x.max(), 7);
        EXPECT_EQ(x.median(), 4);
    }
    {
        shape x(7, 6, 7, 0, 0, 0);
        EXPECT_EQ(x.min(), 6);
        EXPECT_EQ(x.max(), 7);
        EXPECT_EQ(x.median(), 7);
    }
    {
        shape x(6, 6, 7, 0, 0, 0);
        EXPECT_EQ(x.min(), 6);
        EXPECT_EQ(x.max(), 7);
        EXPECT_EQ(x.median(), 6);
    }
}

TEST(apply_gpu, configurtation) {
    using conf = gt::_impl::kernel_configuration;

    gt::uint_t l = 1000;

    {
        gt::uint_t m0 = 1, m1 = 2, m2 = 3;
        gt::uint_t p0 = 2, p1 = 1, p2 = 2;
        gt::uint_t b0 = 1, b1 = 4, b2 = 5;
        gt::uint_t e0 = 67, e1 = 45, e2 = 54;

        gt::array<gt::halo_descriptor, 3> halos{gt::halo_descriptor{m0, p0, b0, e0, l},
            gt::halo_descriptor{m1, p1, b1, e1, l},
            gt::halo_descriptor{m2, p2, b2, e2, l}};

        conf c{halos};

        gt::array<std::size_t, 3> res{67, 50, 3};

        EXPECT_EQ(c.configuration[0], res[0]);
        EXPECT_EQ(c.configuration[1], res[1]);
        EXPECT_EQ(c.configuration[2], res[2]);

        EXPECT_EQ(c.shape(0, 0, 0).start(0), 0);
        EXPECT_EQ(c.shape(0, 0, 0).start(1), 2);
        EXPECT_EQ(c.shape(0, 0, 0).start(2), 2);

        EXPECT_EQ(c.shape(1, 0, 0).start(0), 1);
        EXPECT_EQ(c.shape(1, 0, 0).start(1), 2);
        EXPECT_EQ(c.shape(1, 0, 0).start(2), 2);

        EXPECT_EQ(c.shape(2, 0, 0).start(0), 68);
        EXPECT_EQ(c.shape(2, 0, 0).start(1), 2);
        EXPECT_EQ(c.shape(2, 0, 0).start(2), 2);

        EXPECT_EQ(c.shape(0, 1, 0).start(0), 0);
        EXPECT_EQ(c.shape(0, 1, 0).start(1), 4);
        EXPECT_EQ(c.shape(0, 1, 0).start(2), 2);

        EXPECT_EQ(c.shape(0, 2, 0).start(0), 0);
        EXPECT_EQ(c.shape(0, 2, 0).start(1), 46);
        EXPECT_EQ(c.shape(0, 2, 0).start(2), 2);

        EXPECT_EQ(c.shape(0, 0, 1).start(0), 0);
        EXPECT_EQ(c.shape(0, 0, 1).start(1), 2);
        EXPECT_EQ(c.shape(0, 0, 1).start(2), 5);

        EXPECT_EQ(c.shape(0, 0, 2).start(0), 0);
        EXPECT_EQ(c.shape(0, 0, 2).start(1), 2);
        EXPECT_EQ(c.shape(0, 0, 2).start(2), 55);
    }

    {
        gt::uint_t m0 = 1, m1 = 2, m2 = 3;
        gt::uint_t p0 = 2, p1 = 5, p2 = 3;
        gt::uint_t b0 = 1, b1 = 4, b2 = 5;
        gt::uint_t e0 = 2, e1 = 5, e2 = 5;

        gt::array<gt::halo_descriptor, 3> halos{gt::halo_descriptor{m0, p0, b0, e0, l},
            gt::halo_descriptor{m1, p1, b1, e1, l},
            gt::halo_descriptor{m2, p2, b2, e2, l}};

        conf c{halos};

        gt::array<std::size_t, 3> res{5, 3, 2};

        EXPECT_EQ(c.configuration[0], res[0]);
        EXPECT_EQ(c.configuration[1], res[1]);
        EXPECT_EQ(c.configuration[2], res[2]);
    }

    {
        gt::uint_t m0 = 1, m1 = 1, m2 = 1;
        gt::uint_t p0 = 1, p1 = 1, p2 = 1;
        gt::uint_t b0 = 1, b1 = 1, b2 = 1;
        gt::uint_t e0 = 3, e1 = 3, e2 = 3;

        gt::array<gt::halo_descriptor, 3> halos{gt::halo_descriptor{m0, p0, b0, e0, l},
            gt::halo_descriptor{m1, p1, b1, e1, l},
            gt::halo_descriptor{m2, p2, b2, e2, l}};

        conf c{halos};

        gt::array<std::size_t, 3> res{3, 3, 1};

        EXPECT_EQ(c.configuration[0], res[0]);
        EXPECT_EQ(c.configuration[1], res[1]);
        EXPECT_EQ(c.configuration[2], res[2]);
    }
}

/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/tools/regression_fixture_impl.hpp>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>

namespace gridtools {
    namespace _impl {
        uint_t regression_fixture_base::s_d1 = 0;
        uint_t regression_fixture_base::s_d2 = 0;
        uint_t regression_fixture_base::s_d3 = 0;
        uint_t regression_fixture_base::s_steps = 0;
        bool regression_fixture_base::s_needs_verification = true;

        void regression_fixture_base::flush_cache() {
            static std::size_t n = 1024 * 1024 * 21 / 2;
            static std::vector<double> a_(n), b_(n), c_(n);
            double *a = a_.data();
            double *b = b_.data();
            double *c = c_.data();
#pragma omp parallel for
            for (int i = 0; i < n; i++)
                a[i] = b[i] * c[i];
        }

        void regression_fixture_base::init(int argc, char **argv) {
            if (argc < 4) {
                std::cerr << "Usage: " << argv[0] << " "
                          << "dimx dimy dimz tsteps\n\twhere args are integer sizes of the data fields and tsteps "
                             "is the number of time steps to run in a benchmark run"
                          << std::endl;
                exit(1);
            }
            s_d1 = std::atoi(argv[1]);
            s_d2 = std::atoi(argv[2]);
            s_d3 = std::atoi(argv[3]);
            s_steps = argc > 4 ? std::atoi(argv[4]) : 0;
            s_needs_verification = argc < 6 || std::strcmp(argv[5], "-d") != 0;
        }
    } // namespace _impl
} // namespace gridtools

int main(int argc, char **argv) {
    // Pass command line arguments to googltest
    ::testing::InitGoogleTest(&argc, argv);
    gridtools::_impl::regression_fixture_base::init(argc, argv);
    return RUN_ALL_TESTS();
}

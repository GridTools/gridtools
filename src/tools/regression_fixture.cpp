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
#include <gridtools/tools/regression_fixture_impl.hpp>

#include <cstdlib>
#include <iostream>

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
#ifndef __NVCC__
            static auto constexpr n = 1024 * 1024 * 21 / 2;
            static double a[n];
            static double b[n];
            static double c[n];
            int i;
#pragma omp parallel for private(i)
            for (i = 0; i != n; i++)
                a[i] = b[i] * c[i];
#endif
        }

        void regression_fixture_base::init(int argc, char **argv) {
            if (argc < 4) {
                std::cerr << "Usage: " << argv[0]
                          << "dimx dimy dimz tsteps\n\twhere args are integer sizes of the data fields and tsteps "
                             "is the number of time steps to run in a benchmark run"
                          << std::endl;
                exit(1);
            }
            s_d1 = std::atoi(argv[1]);
            s_d2 = std::atoi(argv[2]);
            s_d3 = std::atoi(argv[3]);
            s_steps = argc > 4 ? std::atoi(argv[4]) : 0;
            s_needs_verification = argc < 5 || std::strcmp(argv[5], "-d") != 0;
        }
    } // namespace _impl
} // namespace gridtools

int main(int argc, char **argv) {
    // Pass command line arguments to googltest
    ::testing::InitGoogleTest(&argc, argv);
    gridtools::_impl::regression_fixture_base::init(argc, argv);
    return RUN_ALL_TESTS();
}

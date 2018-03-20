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
#include "gtest/gtest.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <gridtools.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include <tools/verifier.hpp>
#include "Options.hpp"
#include "backend_select.hpp"

using namespace gridtools;
using namespace enumtype;

/**
   @brief structure containing the Laplacian-specific information.

Contains the stencil operators that compose the multistage stencil in this test
*/
struct lap_function {

    typedef accessor< 0, inout, extent<>, 3 > out_acc;

    typedef accessor< 1, in, extent< -1, 1, -1, 1 >, 3 > in_acc;

    typedef boost::mpl::vector< out_acc, in_acc > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {

        eval(out_acc()) = 4 * eval(in_acc()) - (eval(in_acc(1, 0, 0)) + eval(in_acc(0, 1, 0)) + eval(in_acc(-1, 0, 0)) +
                                                   eval(in_acc(0, -1, 0)));
    }
};

/*!
 * @brief This operator is used for debugging only
 */

std::ostream &operator<<(std::ostream &s, lap_function const) { return s << "lap_function"; }

TEST(Laplace, test) {

    uint_t d1 = Options::getInstance().m_size[0];
    uint_t d2 = Options::getInstance().m_size[1];
    uint_t d3 = Options::getInstance().m_size[2];

    uint_t halo_size = 2;

    /**
       - definition of the storage type, depending on the backend_t which is set as a macro. \todo find another strategy
       for the backend (policy pattern)?
    */
    typedef backend_t::storage_traits_t::storage_info_t< 0, 3 > storage_info_t;
    typedef backend_t::storage_traits_t::data_store_t< float_type, storage_info_t > storage_t;

    /**
        - Instantiation of the actual data fields that are used for input/output
    */
    storage_info_t metadata_(d1, d2, d3);
    storage_t in(metadata_, -1.);
    storage_t out(metadata_, -7.3);

    /**
       - Definition of placeholders. The order of them reflect the order the user will deal with them
       especially the non-temporary ones, in the construction of the domain.
       A placeholder only contains a static const index and a storage type
    */
    typedef arg< 0, storage_t > p_in;
    typedef arg< 1, storage_t > p_out;
    typedef boost::mpl::vector< p_in, p_out > accessor_list;

    /**
       - Construction of the domain. The domain is the physical domain of the problem, with all the physical fields that
       are used, temporary and not
       It must be noted that the only fields to be passed to the constructor are the non-temporary.
       The order in which they have to be passed is the order in which they appear scanning the placeholders in order
       (i.e. the order in the accessor_list?). \todo (I don't particularly like this).
       \note aggregator_type implements the CRTP pattern in order to do static polymorphism (?) Because all what is
       'clonable to gpu' must derive from the CRTP base class.
    */
    aggregator_type< accessor_list > domain(in, out);

    /**
       - Definition of the physical dimensions of the problem.
       The grid constructor takes the horizontal plane dimensions,
       while the vertical ones are set according the the axis property soon after
    */
    halo_descriptor di{halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
    halo_descriptor dj{halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};

    auto grid = make_grid(di, dj, d3);

    auto laplace = make_computation< backend_t >(
        domain, grid, make_multistage(execute< forward >(), make_stage< lap_function >(p_out(), p_in())));

    laplace->ready();

    laplace->steady();

    /**
       Call to gridtools::intermediate::run, which calls Backend::run, does the actual stencil operations on the
       backend.
     */
    laplace->run();

    laplace->finalize();

    storage_t ref(metadata_, -7.3);

    auto refv = make_host_view(ref);
    auto inv = make_host_view(in);
    for (uint_t i = halo_size; i != d1 - halo_size; ++i) {
        for (uint_t j = halo_size; j != d2 - halo_size; ++j) {
            for (uint_t k = 0; k != d3; ++k) {
                refv(i, j, k) =
                    4 * inv(i, j, k) - (inv(i + 1, j, k) + inv(i, j + 1, k) + inv(i - 1, j, k) + inv(i, j - 1, k));
            }
        }
    }

#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-12);
#endif
    array< array< uint_t, 2 >, 3 > halos{{{halo_size, halo_size}, {halo_size, halo_size}, {0, 0}}};
    bool result = verif.verify(grid, ref, out, halos);

#ifdef BENCHMARK
    std::cout << laplace->print_meter() << std::endl;
#endif

    ASSERT_TRUE(result);
}

int main(int argc, char **argv) {
    // Pass command line arguments to googltest
    ::testing::InitGoogleTest(&argc, argv);

    if (argc != 4) {
        printf("Usage: laplace_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields\n");
        return 1;
    }

    for (int i = 0; i != 3; ++i) {
        Options::getInstance().m_size[i] = atoi(argv[i + 1]);
    }

    return RUN_ALL_TESTS();
}

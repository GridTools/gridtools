/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#pragma once

#include <stencil-composition/stencil-composition.hpp>
#include "benchmarker.hpp"

#ifdef __CUDACC__
typedef gridtools::halo< 2, 0, 0 > halo_t;
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
typedef gridtools::halo< 0, 0, 2 > halo_t;
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
typedef gridtools::halo< 0, 2, 0 > halo_t;
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

/**
  @file
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend,
  in which a misaligned storage is aligned
*/

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

namespace aligned_copy_stencil {

    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    typedef BACKEND::storage_traits_t::storage_info_t< 0, 3, halo_t > meta_data_t;
    typedef BACKEND::storage_traits_t::data_store_t< float_type, meta_data_t > storage_t;

    // These are the stencil operators that compose the multistage stencil in this test
    struct copy_functor {

#ifdef __CUDACC__
        /** @brief checking all storages alignment using a specific storage_info

            \param storage_id ordinal number identifying the storage_info checked
            \param boundary ordinal number identifying the alignment
        */
        template < unsigned I, typename ItDomain >
        GT_FUNCTION static bool check_pointer_alignment(ItDomain const &it_domain, uint_t boundary) {
            bool result_ = true;
            if (threadIdx.x == 0) {
                auto ptr = (static_cast<float_type*>(it_domain.get().data_pointer().template get<I>()[0])+it_domain.get().index()[0]);
                result_ = (((unsigned long)ptr & (boundary-1)) == 0);
            }
            return result_;
        }
#endif

        typedef accessor< 0, enumtype::in, extent< 0, 0, 0, 0 >, 3 > in;
        typedef accessor< 1, enumtype::inout, extent< 0, 0, 0, 0 >, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {

#ifdef __CUDACC__
#ifndef NDEBUG
            if (!check_pointer_alignment<0>(eval, meta_data_t::alignment_t::value) || 
                !check_pointer_alignment<1>(eval, meta_data_t::alignment_t::value)) {
                printf("alignment error in some storages with first meta_storage \n");
                assert(false);
            }
#endif
#endif
            eval(out()) = eval(in());
        }
    };

    bool test(uint_t d1, uint_t d2, uint_t d3) {

        meta_data_t meta_data_(d1, d2, d3);

        // Definition of the actual data fields that are used for input/output
        storage_t in(meta_data_, (float_type)0.);
        storage_t out(meta_data_, (float_type)-1.);
        
        auto inv = make_host_view(in);
        auto outv = make_host_view(out);
        for (uint_t i = halo_t::at< 0 >(); i < d1 + halo_t::at< 0 >(); ++i)
            for (uint_t j = halo_t::at< 1 >(); j < d2 + halo_t::at< 1 >(); ++j)
                for (uint_t k = halo_t::at< 2 >(); k < d3 + halo_t::at< 2 >(); ++k) {
                    inv(i, j, k) = i + j + k;
                }

        typedef arg< 0, storage_t > p_in;
        typedef arg< 1, storage_t > p_out;

        typedef boost::mpl::vector< p_in, p_out > accessor_list;
        // construction of the domain. The domain is the physical domain of the problem, with all the physical fields
        // that are used, temporary and not
        // It must be noted that the only fields to be passed to the constructor are the non-temporary.
        // The order in which they have to be passed is the order in which they appear scanning the placeholders in
        // order. (I don't particularly like this)
        gridtools::aggregator_type< accessor_list > domain(in, out);

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        // gridtools::coordinates<axis> grid(2,d1-2,2,d2-2);
        uint_t di[5] = {
            halo_t::at< 0 >(), 0, halo_t::at< 0 >(), d1 + halo_t::at< 0 >() - 1, d1 + halo_t::at< 0 >()};
        uint_t dj[5] = {
            halo_t::at< 1 >(), 0, halo_t::at< 1 >(), d2 + halo_t::at< 1 >() - 1, d2 + halo_t::at< 1 >()};

        gridtools::grid< axis > grid(di, dj);

        grid.value_list[0] = halo_t::at< 2 >();
        grid.value_list[1] = d3 + halo_t::at< 2 >() - 1;

        auto copy = gridtools::make_computation< gridtools::BACKEND >(domain, grid,
                gridtools::make_multistage(execute< forward >(), gridtools::make_stage< copy_functor >(p_in(), p_out()))
        );

        copy->ready();
        copy->steady();
        copy->run();
        copy->finalize();

#ifdef BENCHMARK
        std::cout << copy->print_meter() << std::endl;
#endif

        bool success = true;
        for (uint_t i = halo_t::at< 0 >(); i < d1 + halo_t::at< 0 >(); ++i)
            for (uint_t j = halo_t::at< 1 >(); j < d2 + halo_t::at< 1 >(); ++j)
                for (uint_t k = halo_t::at< 2 >(); k < d3 + halo_t::at< 2 >(); ++k) {
                    if (inv(i, j, k) != outv(i, j, k) || outv(i, j, k) != i + j + k) {
                        std::cout << "error in " << i << ", " << j << ", " << k << ": "
                                  << "in = " << inv(i, j, k) << ", out = " << outv(i, j, k) << std::endl;
                        success = false;
                    }
                }
        return success;
    }
} // namespace aligned_copy_stencil

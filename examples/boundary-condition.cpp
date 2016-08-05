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
#include <gridtools.hpp>
#include <common/halo_descriptor.hpp>

#ifdef CUDA_EXAMPLE
#include <boundary-conditions/apply_gpu.hpp>
#else
#include <boundary-conditions/apply.hpp>
#endif

using gridtools::direction;
using gridtools::sign;
using gridtools::minus_;
using gridtools::zero_;
using gridtools::plus_;

#include <stencil-composition/stencil-composition.hpp>

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>

using namespace gridtools;
using namespace enumtype;

#ifdef CUDA_EXAMPLE
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif


template <typename T>
struct direction_bc_input {
    T value;

    GT_FUNCTION
    direction_bc_input()
        : value(1)
    {}

    GT_FUNCTION
    direction_bc_input(T v)
        : value(v)
    {}

    // relative coordinates
    template <typename Direction, typename DataField0, typename DataField1>
    GT_FUNCTION
    void operator()(Direction,
                    DataField0 & data_field0, DataField1 const & data_field1,
                    uint_t i, uint_t j, uint_t k) const {
        data_field0(i,j,k) = data_field1(i,j,k) * value;
    }

    // relative coordinates
    template <sign I, sign K, typename DataField0, typename DataField1>
    GT_FUNCTION
    void operator()(direction<I, minus_, K>,
                    DataField0 & data_field0, DataField1 const & data_field1,
                    uint_t i, uint_t j, uint_t k) const {
        data_field0(i,j,k) = 88 * value;
    }

    // relative coordinates
    template <sign K, typename DataField0, typename DataField1>
    GT_FUNCTION
    void operator()(direction<minus_, minus_, K>,
                    DataField0 & data_field0, DataField1 const & data_field1,
                    uint_t i, uint_t j, uint_t k) const {
        data_field0(i,j,k) = 77777 * value;
    }

    template <typename DataField0, typename DataField1>
    GT_FUNCTION
    void operator()(direction<minus_, minus_, minus_>,
                    DataField0 & data_field0, DataField1 const & data_field1,
                    uint_t i, uint_t j, uint_t k) const {
        data_field0(i,j,k) = 55555 * value;
    }
};



int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " dimx dimy dimz\n"
               " where args are integer sizes of the data fields" << std::endl;
        return EXIT_FAILURE;
    }

    uint_t d1 = atoi(argv[1]);
    uint_t d2 = atoi(argv[2]);
    uint_t d3 = atoi(argv[3]);

    typedef BACKEND::storage_type<int_t, BACKEND::storage_info<0,gridtools::layout_map<0,1,2> > >::type storage_type;

    // Definition of the actual data fields that are used for input/output
    storage_type::storage_info_type meta_(d1,d2,d3);
    storage_type in(meta_, "in");
    in.initialize(-1);
    storage_type out(meta_, "out");
    out.initialize(-7);
    storage_type coeff(meta_, "coeff");
    coeff.initialize(8);

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                in(i,j,k) = 0;
                out(i,j,k) = i+j+k;
            }
        }
    }

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1,1,1,d1-2,d1);
    halos[1] = gridtools::halo_descriptor(1,1,1,d2-2,d2);
    halos[2] = gridtools::halo_descriptor(1,1,1,d3-2,d3);

#ifdef CUDA_EXAMPLE
    //TODO also metadata must be copied/used here
    in.h2d_update();
    out.h2d_update();

    gridtools::boundary_apply_gpu<direction_bc_input<uint_t> >(halos, direction_bc_input<uint_t>(2)).apply(in, out);

    in.d2h_update();
#else
    gridtools::boundary_apply<direction_bc_input<uint_t> >(halos, direction_bc_input<uint_t>(2)).apply(in, out);
#endif

}

/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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

#include <stencil_composition/stencil_composition.hpp>

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

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

#include "gtest/gtest.h"

#include "common/halo_descriptor.hpp"
#include <gridtools.hpp>

#ifdef __CUDACC__
#include <boundary-conditions/apply_gpu.hpp>
#else
#include <boundary-conditions/apply.hpp>
#endif

#include <boundary-conditions/copy.hpp>
#include <boundary-conditions/value.hpp>
#include <boundary-conditions/zero.hpp>

using gridtools::direction;
using gridtools::sign;
using gridtools::minus_;
using gridtools::zero_;
using gridtools::plus_;

#include "stencil-composition/stencil-composition.hpp"

#include <stdio.h>
#include <stdlib.h>

#include <boost/utility/enable_if.hpp>

using namespace gridtools;
using namespace enumtype;

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

struct bc_basic {

    // relative coordinates
    template < typename Direction, typename DataField0 >
    GT_FUNCTION void operator()(Direction, DataField0 &data_field0, uint_t i, uint_t j, uint_t k) const {
        data_field0(i, j, k) = i + j + k;
    }
};

#define SET_TO_ZERO                                                                            \
    template < typename Direction, typename DataField0 >                                       \
    void operator()(Direction, DataField0 & data_field0, uint_t i, uint_t j, uint_t k) const { \
        data_field0(i, j, k) = 0;                                                              \
    }

template < sign X >
struct is_minus {
    static const bool value = (X == minus_);
};

template < typename T, typename U >
struct is_one_of {
    static const bool value = T::value || U::value;
};

struct bc_two {

    template < typename Direction, typename DataField0 >
    GT_FUNCTION void operator()(Direction, DataField0 &data_field0, uint_t i, uint_t j, uint_t k) const {
        data_field0(i, j, k) = 0;
    }

    template < sign I, sign J, sign K, typename DataField0 >
    GT_FUNCTION void operator()(direction< I, J, K >,
        DataField0 &data_field0,
        uint_t i,
        uint_t j,
        uint_t k,
        typename boost::enable_if< is_one_of< is_minus< J >, is_minus< K > > >::type *dummy = 0) const {
        data_field0(i, j, k) = (i + j + k + 1);
    }

    // THE CODE ABOVE IS A REPLACEMENT OF THE FOLLOWING 4 DIFFERENT SPECIALIZATIONS
    // IT IS UGLY BUT CAN SAVE QUITE A BIT OF CODE

    // template <sign I, sign K, typename DataField0>
    // void operator()(direction<I,minus,K>,
    //                 DataField0 & data_field0,
    //                 int i, int j, int k) const {
    //     data_field0(i,j,k) = i+j+k+1;
    // }

    // template <sign J, sign K, typename DataField0>
    // void operator()(direction<minus,J,K>,
    //                 DataField0 & data_field0,
    //                 int i, int j, int k) const {
    //     data_field0(i,j,k) = i+j+k+1;
    // }

    // template <sign I, typename DataField0>
    // void operator()(direction<I,minus,minus>,
    //                 DataField0 & data_field0,
    //                 int i, int j, int k) const {
    //     data_field0(i,j,k) = i+j+k+1;
    // }

    // template <typename DataField0>
    // void operator()(direction<minus,minus,minus>,
    //                 DataField0 & data_field0,
    //                 int i, int j, int k) const {
    //     data_field0(i,j,k) = i+j+k+1;
    // }
};

struct minus_predicate {
    template < sign I, sign J, sign K >
    bool operator()(direction< I, J, K >) const {
        if (I == minus_ || J == minus_ || K == minus_)
            return false;
        return true;
    }
};

bool basic() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    typedef gridtools::BACKEND::storage_type< int_t,
        gridtools::BACKEND::storage_info< 0, layout_map< 0, 1, 2 > > >::type storage_type;

    // Definition of the actual data fields that are used for input/output

    gridtools::BACKEND::storage_info< 0, layout_map< 0, 1, 2 > > meta_(d1, d2, d3);
    storage_type in(meta_, 0., "in");

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                in(i, j, k) = 0;
            }
        }
    }

    gridtools::array< gridtools::halo_descriptor, 3 > halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

#ifdef __CUDACC__
    in.h2d_update();
    in.clone_to_device();

    gridtools::boundary_apply_gpu< bc_basic >(halos, bc_basic()).apply(in);

    in.d2h_update();
#else
    gridtools::boundary_apply< bc_basic >(halos, bc_basic()).apply(in);
#endif

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (in(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (in(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    return result;
}

bool predicate() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    typedef gridtools::BACKEND::storage_type< int_t,
        gridtools::BACKEND::storage_info< 0, layout_map< 0, 1, 2 > > >::type storage_type;

    // Definition of the actual data fields that are used for input/output

    gridtools::BACKEND::storage_info< 0, layout_map< 0, 1, 2 > > meta_(d1, d2, d3);
    storage_type in(meta_, -1, "in");

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                in(i, j, k) = 0;
            }
        }
    }

    gridtools::array< gridtools::halo_descriptor, 3 > halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

#ifdef __CUDACC__
    in.h2d_update();
    in.clone_to_device();

    gridtools::boundary_apply_gpu< bc_basic, minus_predicate >(halos, bc_basic(), minus_predicate()).apply(in);

    in.d2h_update();
#else
    gridtools::boundary_apply< bc_basic, minus_predicate >(halos, bc_basic(), minus_predicate()).apply(in);
#endif

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1; ++i) {
        for (uint_t j = 1; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (in(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 1; k < d3; ++k) {
                if (in(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 1; j < d2; ++j) {
            for (uint_t k = 1; k < d3; ++k) {
                if (in(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    return result;
}

bool twosurfaces() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    typedef gridtools::BACKEND::storage_type< int_t,
        gridtools::BACKEND::storage_info< 0, layout_map< 0, 1, 2 > > >::type storage_type;

    // Definition of the actual data fields that are used for input/output

    gridtools::BACKEND::storage_info< 0, layout_map< 0, 1, 2 > > meta_(d1, d2, d3);
    storage_type in(meta_, -1, "in");

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                in(i, j, k) = 1;
            }
        }
    }

    gridtools::array< gridtools::halo_descriptor, 3 > halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

#ifdef __CUDACC__
    in.h2d_update();
    in.clone_to_device();

    gridtools::boundary_apply_gpu< bc_two >(halos, bc_two()).apply(in);

    in.d2h_update();
#else
    gridtools::boundary_apply< bc_two >(halos, bc_two()).apply(in);
#endif

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (in(i, j, k) != i + j + k + 1) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 1; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != i + j + k + 1) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 1; k < d3; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 1; j < d2; ++j) {
            for (uint_t k = 1; k < d3; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 1; j < d2; ++j) {
            for (uint_t k = 1; k < d3; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (in(i, j, k) != 1) {
                    result = false;
                }
            }
        }
    }

    return result;
}

bool usingzero_1() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    typedef gridtools::BACKEND::storage_type< int_t,
        gridtools::BACKEND::storage_info< 0, layout_map< 0, 1, 2 > > >::type storage_type;

    // Definition of the actual data fields that are used for input/output

    gridtools::BACKEND::storage_info< 0, layout_map< 0, 1, 2 > > meta_(d1, d2, d3);
    storage_type in(meta_, -1, "in");

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                in(i, j, k) = -1;
            }
        }
    }

    gridtools::array< gridtools::halo_descriptor, 3 > halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

#ifdef __CUDACC__
    in.h2d_update();
    in.clone_to_device();

    gridtools::boundary_apply_gpu< gridtools::zero_boundary >(halos).apply(in);

    in.d2h_update();
#else
    gridtools::boundary_apply< gridtools::zero_boundary >(halos).apply(in);
#endif

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (in(i, j, k) != -1) {
                    result = false;
                }
            }
        }
    }

    return result;
}

bool usingzero_2() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    typedef gridtools::BACKEND::storage_type< int_t,
        gridtools::BACKEND::storage_info< 0, layout_map< 0, 1, 2 > > >::type storage_type;

    // Definition of the actual data fields that are used for input/output

    gridtools::BACKEND::storage_info< 0, layout_map< 0, 1, 2 > > meta_(d1, d2, d3);
    storage_type in(meta_, -1, "in");
    storage_type out(meta_, -1, "out");

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                in(i, j, k) = -1;
                out(i, j, k) = -1;
            }
        }
    }

    gridtools::array< gridtools::halo_descriptor, 3 > halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

#ifdef __CUDACC__
    in.h2d_update();
    out.h2d_update();
    in.clone_to_device();
    out.clone_to_device();

    gridtools::boundary_apply_gpu< gridtools::zero_boundary >(halos).apply(in, out);

    in.d2h_update();
    out.d2h_update();
#else
    gridtools::boundary_apply< gridtools::zero_boundary >(halos).apply(in, out);
#endif

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
                if (out(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
                if (out(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
                if (out(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
                if (out(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
                if (out(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != 0) {
                    result = false;
                }
                if (out(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (in(i, j, k) != -1) {
                    result = false;
                }
                if (out(i, j, k) != -1) {
                    result = false;
                }
            }
        }
    }

    return result;
}

bool usingvalue_2() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    typedef gridtools::BACKEND::storage_type< int_t,
        gridtools::BACKEND::storage_info< 0, layout_map< 0, 1, 2 > > >::type storage_type;

    // Definition of the actual data fields that are used for input/output

    gridtools::BACKEND::storage_info< 0, layout_map< 0, 1, 2 > > meta_(d1, d2, d3);
    storage_type in(meta_, -1, "in");
    storage_type out(meta_, -1, "out");

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                in(i, j, k) = -1;
                out(i, j, k) = -1;
            }
        }
    }

    gridtools::array< gridtools::halo_descriptor, 3 > halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

#ifdef __CUDACC__
    in.h2d_update();
    out.h2d_update();
    in.clone_to_device();
    out.clone_to_device();

    gridtools::boundary_apply_gpu< gridtools::value_boundary< int_t > >(halos, gridtools::value_boundary< int_t >(101))
        .apply(in, out);

    in.d2h_update();
    out.d2h_update();
#else
    gridtools::boundary_apply< gridtools::value_boundary< int_t > >(halos, gridtools::value_boundary< int_t >(101))
        .apply(in, out);
#endif

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (in(i, j, k) != 101) {
                    result = false;
                }
                if (out(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (in(i, j, k) != 101) {
                    result = false;
                }
                if (out(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != 101) {
                    result = false;
                }
                if (out(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != 101) {
                    result = false;
                }
                if (out(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != 101) {
                    result = false;
                }
                if (out(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != 101) {
                    result = false;
                }
                if (out(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (in(i, j, k) != -1) {
                    result = false;
                }
                if (out(i, j, k) != -1) {
                    result = false;
                }
            }
        }
    }

    return result;
}

bool usingcopy_3() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    typedef gridtools::BACKEND::storage_type< int_t,
        gridtools::BACKEND::storage_info< 0, layout_map< 0, 1, 2 > > >::type storage_type;

    // Definition of the actual data fields that are used for input/output

    gridtools::BACKEND::storage_info< 0, layout_map< 0, 1, 2 > > meta_(d1, d2, d3);
    storage_type src(meta_, -1, "src");
    storage_type one(meta_, -1, "one");
    storage_type two(meta_, -1, "two");

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                src(i, j, k) = i + k + j;
                one(i, j, k) = -1;
                two(i, j, k) = 0;
            }
        }
    }

    gridtools::array< gridtools::halo_descriptor, 3 > halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

#ifdef __CUDACC__
    one.h2d_update();
    one.clone_to_device();
    two.h2d_update();
    two.clone_to_device();
    src.h2d_update();
    src.clone_to_device();

    gridtools::boundary_apply_gpu< gridtools::copy_boundary >(halos).apply(one, two, src);

    one.d2h_update();
    two.d2h_update();
#else
    gridtools::boundary_apply< gridtools::copy_boundary >(halos).apply(one, two, src);
#endif

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (one(i, j, k) != i + j + k) {
                    result = false;
                }
                if (two(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (one(i, j, k) != i + j + k) {
                    result = false;
                }
                if (two(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (one(i, j, k) != i + j + k) {
                    result = false;
                }
                if (two(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (one(i, j, k) != i + j + k) {
                    result = false;
                }
                if (two(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (one(i, j, k) != i + j + k) {
                    result = false;
                }
                if (two(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (one(i, j, k) != i + j + k) {
                    result = false;
                }
                if (two(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (one(i, j, k) != -1) {
                    result = false;
                }
                if (two(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    return result;
}

TEST(boundaryconditions, predicate) { EXPECT_EQ(predicate(), true); }

TEST(boundaryconditions, twosurfaces) { EXPECT_EQ(twosurfaces(), true); }

TEST(boundaryconditions, usingzero_1) { EXPECT_EQ(usingzero_1(), true); }

TEST(boundaryconditions, usingzero_2) { EXPECT_EQ(usingzero_2(), true); }

TEST(boundaryconditions, basic) { EXPECT_EQ(basic(), true); }

TEST(boundaryconditions, usingvalue2) { EXPECT_EQ(usingvalue_2(), true); }

TEST(boundaryconditions, usingcopy3) { EXPECT_EQ(usingcopy_3(), true); }

/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cstdio>
#include <cstdlib>

#include <gtest/gtest.h>

#include <gridtools/boundary_conditions/boundary.hpp>
#include <gridtools/boundary_conditions/copy.hpp>
#include <gridtools/boundary_conditions/value.hpp>
#include <gridtools/boundary_conditions/zero.hpp>
#include <gridtools/common/halo_descriptor.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;

struct bc_basic {

    // relative coordinates
    template <typename Direction, typename DataField0>
    GT_FUNCTION void operator()(Direction, DataField0 &data_field0, uint_t i, uint_t j, uint_t k) const {
        data_field0(i, j, k) = i + j + k;
    }
};

#define SET_TO_ZERO                                                                           \
    template <typename Direction, typename DataField0>                                        \
    void operator()(Direction, DataField0 &data_field0, uint_t i, uint_t j, uint_t k) const { \
        data_field0(i, j, k) = 0;                                                             \
    }

template <sign X>
struct is_minus {
    static const bool value = (X == minus_);
};

template <typename T, typename U>
struct is_one_of {
    static const bool value = T::value || U::value;
};

struct bc_two {

    template <typename Direction, typename DataField0>
    GT_FUNCTION void operator()(Direction, DataField0 &data_field0, uint_t i, uint_t j, uint_t k) const {
        data_field0(i, j, k) = 0;
    }

    template <sign I, sign J, sign K, typename DataField0>
    GT_FUNCTION void operator()(direction<I, J, K>,
        DataField0 &data_field0,
        uint_t i,
        uint_t j,
        uint_t k,
        std::enable_if_t<is_one_of<is_minus<J>, is_minus<K>>::value> * = nullptr) const {
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
    template <sign I, sign J, sign K>
    bool operator()(direction<I, J, K>) const {
        if (I == minus_ || J == minus_ || K == minus_)
            return false;
        else
            return true;
    }
};

bool basic() {

    uint_t d1 = 4;
    uint_t d2 = 3;
    uint_t d3 = 5;

    typedef storage_traits<backend_t>::storage_info_t<0, 3> meta_data_t;
    typedef storage_traits<backend_t>::data_store_t<int_t, meta_data_t> storage_t;

    // Definition of the actual data fields that are used for input/output
    meta_data_t meta_(d1, d2, d3);
    storage_t in(meta_, 0);
    auto inv = make_host_view(in);

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

    in.sync();
    gridtools::boundary<bc_basic, backend_t>(halos, bc_basic()).apply(in);

    in.sync();

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (inv(i, j, k) != 0) {
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

    typedef storage_traits<backend_t>::storage_info_t<0, 3> meta_data_t;
    typedef storage_traits<backend_t>::data_store_t<int_t, meta_data_t> storage_t;

    // Definition of the actual data fields that are used for input/output

    meta_data_t meta_(d1, d2, d3);
    storage_t in(meta_, 0);
    auto inv = make_host_view(in);

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

    in.sync();
#ifdef __CUDACC__
    auto indv = make_target_view(in);
    gridtools::boundary_apply_gpu<bc_basic, minus_predicate>(halos, bc_basic(), minus_predicate()).apply(indv);
#else
    gridtools::boundary_apply<bc_basic, minus_predicate>(halos, bc_basic(), minus_predicate()).apply(inv);
#endif
    in.sync();

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1; ++i) {
        for (uint_t j = 1; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 1; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 1; j < d2; ++j) {
            for (uint_t k = 1; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (inv(i, j, k) != 0) {
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

    typedef storage_traits<backend_t>::storage_info_t<0, 3> meta_data_t;
    typedef storage_traits<backend_t>::data_store_t<int_t, meta_data_t> storage_t;

    // Definition of the actual data fields that are used for input/output

    meta_data_t meta_(d1, d2, d3);
    storage_t in(meta_, 1);
    auto inv = make_host_view(in);

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

    in.sync();
#ifdef __CUDACC__
    auto indv = make_target_view(in);
    gridtools::boundary_apply_gpu<bc_two>(halos, bc_two()).apply(indv);
#else
    gridtools::boundary_apply<bc_two>(halos, bc_two()).apply(inv);
#endif
    in.sync();

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (inv(i, j, k) != i + j + k + 1) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 1; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k + 1) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 1; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 1; j < d2; ++j) {
            for (uint_t k = 1; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 1; j < d2; ++j) {
            for (uint_t k = 1; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (inv(i, j, k) != 1) {
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

    typedef storage_traits<backend_t>::storage_info_t<0, 3> meta_data_t;
    typedef storage_traits<backend_t>::data_store_t<int_t, meta_data_t> storage_t;

    // Definition of the actual data fields that are used for input/output

    meta_data_t meta_(d1, d2, d3);
    storage_t in(meta_, -1);
    auto inv = make_host_view(in);

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

    in.sync();
#ifdef __CUDACC__
    auto indv = make_target_view(in);
    gridtools::boundary_apply_gpu<gridtools::zero_boundary>(halos).apply(indv);
#else
    gridtools::boundary_apply<gridtools::zero_boundary>(halos).apply(inv);
#endif
    in.sync();

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (inv(i, j, k) != -1) {
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

    typedef storage_traits<backend_t>::storage_info_t<0, 3> meta_data_t;
    typedef storage_traits<backend_t>::data_store_t<int_t, meta_data_t> storage_t;

    // Definition of the actual data fields that are used for input/output

    meta_data_t meta_(d1, d2, d3);
    storage_t in(meta_, -1);
    storage_t out(meta_, -1);
    auto inv = make_host_view(in);
    auto outv = make_host_view(out);

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

    in.sync();
    out.sync();
#ifdef __CUDACC__
    auto indv = make_target_view(in);
    auto outdv = make_target_view(out);
    gridtools::boundary_apply_gpu<gridtools::zero_boundary>(halos).apply(indv, outdv);
#else
    gridtools::boundary_apply<gridtools::zero_boundary>(halos).apply(inv, outv);
#endif
    in.sync();
    out.sync();

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
                if (outv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
                if (outv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
                if (outv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
                if (outv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
                if (outv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
                if (outv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (inv(i, j, k) != -1) {
                    result = false;
                }
                if (outv(i, j, k) != -1) {
                    result = false;
                }
            }
        }
    }

    return result;
}

bool usingzero_3_empty_halos() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    typedef storage_traits<backend_t>::storage_info_t<0, 3> meta_data_t;
    typedef storage_traits<backend_t>::data_store_t<int_t, meta_data_t> storage_t;

    // Definition of the actual data fields that are used for input/output

    meta_data_t meta_(d1, d2, d3);
    storage_t in(meta_, -1);
    storage_t out(meta_, -1);
    auto inv = make_host_view(in);
    auto outv = make_host_view(out);

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(0, 0, 0, d2 - 1, d2);
    halos[2] = gridtools::halo_descriptor(0, 0, 0, d3 - 1, d3);

    in.sync();
    out.sync();
#ifdef __CUDACC__
    auto indv = make_target_view(in);
    auto outdv = make_target_view(out);
    gridtools::boundary_apply_gpu<gridtools::zero_boundary>(halos).apply(indv, outdv);
#else
    gridtools::boundary_apply<gridtools::zero_boundary>(halos).apply(inv, outv);
#endif
    in.sync();
    out.sync();

    bool result = true;

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
                if (outv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
                if (outv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != -1) {
                    result = false;
                }
                if (outv(i, j, k) != -1) {
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

    typedef storage_traits<backend_t>::storage_info_t<0, 3> meta_data_t;
    typedef storage_traits<backend_t>::data_store_t<int_t, meta_data_t> storage_t;

    // Definition of the actual data fields that are used for input/output

    meta_data_t meta_(d1, d2, d3);
    storage_t in(meta_, -1);
    storage_t out(meta_, -1);
    auto inv = make_host_view(in);
    auto outv = make_host_view(out);

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

    in.sync();
    out.sync();
#ifdef __CUDACC__
    auto indv = make_target_view(in);
    auto outdv = make_target_view(out);
    gridtools::boundary_apply_gpu<gridtools::value_boundary<int_t>>(halos, gridtools::value_boundary<int_t>(101))
        .apply(indv, outdv);
#else
    gridtools::boundary_apply<gridtools::value_boundary<int_t>>(halos, gridtools::value_boundary<int_t>(101))
        .apply(inv, outv);
#endif
    in.sync();
    out.sync();

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (inv(i, j, k) != 101) {
                    result = false;
                }
                if (outv(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (inv(i, j, k) != 101) {
                    result = false;
                }
                if (outv(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 101) {
                    result = false;
                }
                if (outv(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 101) {
                    result = false;
                }
                if (outv(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 101) {
                    result = false;
                }
                if (outv(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 101) {
                    result = false;
                }
                if (outv(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (inv(i, j, k) != -1) {
                    result = false;
                }
                if (outv(i, j, k) != -1) {
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

    typedef storage_traits<backend_t>::storage_info_t<0, 3> meta_data_t;
    typedef storage_traits<backend_t>::data_store_t<int_t, meta_data_t> storage_t;

    // Definition of the actual data fields that are used for input/output

    meta_data_t meta_(d1, d2, d3);
    storage_t src(meta_, -1);
    storage_t one(meta_, -1);
    storage_t two(meta_, 0);

    auto srcv = make_host_view(src);
    auto onev = make_host_view(one);
    auto twov = make_host_view(two);

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                srcv(i, j, k) = i + k + j;
            }
        }
    }

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

    src.sync();
    one.sync();
    two.sync();
#ifdef __CUDACC__
    auto srcdv = make_target_view(src);
    auto onedv = make_target_view(one);
    auto twodv = make_target_view(two);
    gridtools::boundary_apply_gpu<gridtools::copy_boundary>(halos).apply(onedv, twodv, srcdv);
#else
    gridtools::boundary_apply<gridtools::copy_boundary>(halos).apply(onev, twov, srcv);
#endif
    src.sync();
    one.sync();
    two.sync();

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (onev(i, j, k) != i + j + k) {
                    result = false;
                }
                if (twov(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (onev(i, j, k) != i + j + k) {
                    result = false;
                }
                if (twov(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (onev(i, j, k) != i + j + k) {
                    result = false;
                }
                if (twov(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (onev(i, j, k) != i + j + k) {
                    result = false;
                }
                if (twov(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (onev(i, j, k) != i + j + k) {
                    result = false;
                }
                if (twov(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (onev(i, j, k) != i + j + k) {
                    result = false;
                }
                if (twov(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (onev(i, j, k) != -1) {
                    result = false;
                }
                if (twov(i, j, k) != 0) {
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

TEST(boundaryconditions, usingzero_3_empty_halos) { EXPECT_EQ(usingzero_3_empty_halos(), true); }

TEST(boundaryconditions, basic) { EXPECT_EQ(basic(), true); }

TEST(boundaryconditions, usingvalue2) { EXPECT_EQ(usingvalue_2(), true); }

TEST(boundaryconditions, usingcopy3) { EXPECT_EQ(usingcopy_3(), true); }

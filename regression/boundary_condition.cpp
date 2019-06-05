/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <iostream>

#include <gridtools/boundary_conditions/boundary.hpp>
#include <gridtools/common/halo_descriptor.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;

template <typename T>
struct direction_bc_input {
    T value;

    GT_FUNCTION
    direction_bc_input() : value(1) {}

    GT_FUNCTION
    direction_bc_input(T v) : value(v) {}

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
    GT_FUNCTION void operator()(
        direction<minus_, minus_, K>, DataField0 &, DataField1 const &data_field1, uint_t i, uint_t j, uint_t k) const {
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

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0]
                  << " dimx dimy dimz\n"
                     " where args are integer sizes of the data fields"
                  << std::endl;
        return EXIT_FAILURE;
    }

    uint_t d1 = atoi(argv[1]);
    uint_t d2 = atoi(argv[2]);
    uint_t d3 = atoi(argv[3]);

    typedef storage_traits<backend_t>::storage_info_t<0, 3, halo<1, 1, 1>> meta_data_t;
    typedef storage_traits<backend_t>::data_store_t<int_t, meta_data_t> storage_t;

    // Definition of the actual data fields that are used for input/output
    meta_data_t meta_(d1, d2, d3);
    storage_t in_s(meta_, [](int i, int j, int k) { return i + j + k; }, "in");
    storage_t out_s(meta_, 0, "out");

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

    // sync the data stores if needed
    in_s.sync();
    out_s.sync();

    gridtools::template boundary<direction_bc_input<uint_t>, backend_t>(halos, direction_bc_input<uint_t>(2))
        .apply(in_s, out_s);

    // sync the data stores if needed
    in_s.sync();
    out_s.sync();

    // making the views to access and check correctness
    auto in = make_host_view(in_s);
    auto out = make_host_view(out_s);

    assert(check_consistency(in_s, in) && "view is in an inconsistent state.");
    assert(check_consistency(out_s, out) && "view is in an inconsistent state.");

    // reactivate views and check consistency
    in_s.reactivate_host_write_views();
    out_s.reactivate_host_write_views();
    assert(check_consistency(in_s, in) && "view is in an inconsistent state.");
    assert(check_consistency(out_s, out) && "view is in an inconsistent state.");

    // check inner domain (should be zero)
    bool error = false;
    for (uint_t i = 1; i < d3 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d1 - 1; ++k) {
                if (in(k, j, i) != i + j + k) {
                    std::cout << "Error: INPUT field got modified " << k << " " << j << " " << i << "\n";
                    error = true;
                }
                if (out(k, j, i) != 0) {
                    std::cout << "Error: Inner domain of OUTPUT field got modified " << k << " " << j << " " << i
                              << "\n";
                    error = true;
                }
            }
        }
    }

    // check edge column
    if (out(0, 0, 0) != 111110) {
        std::cout << "Error: edge column values in OUTPUT field are wrong 0 0 0\n";
        error = true;
    }
    for (uint_t k = 1; k < d3; ++k) {
        if (out(0, 0, k) != 155554) {
            std::cout << "Error: edge column values in OUTPUT field are wrong 0 0 " << k << "\n";
            error = true;
        }
    }

    // check j==0 i>0 surface
    for (uint_t i = 1; i < d1; ++i) {
        for (uint_t k = 0; k < d3; ++k) {
            if (out(i, 0, k) != 176) {
                std::cout << "Error: j==0 surface values in OUTPUT field are wrong " << i << " 0 " << k << "\n";
                error = true;
            }
        }
    }

    // check outer domain
    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                // check outer surfaces of the cube
                if (((i == 0 || i == d1 - 1) && j > 0) || (j > 0 && (k == 0 || k == d3 - 1))) {
                    if (out(i, j, k) != in(i, j, k) * 2) {
                        std::cout << "Error: values in OUTPUT field are wrong " << i << " " << j << " " << k << "\n";
                        error = true;
                    }
                }
            }
        }
    }

    if (error) {
        std::cout << "TEST failed.\n";
        abort();
    }

    return error;
}

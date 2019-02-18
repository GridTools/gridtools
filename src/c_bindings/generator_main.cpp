/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <fstream>
#include <iostream>

#include <gridtools/c_bindings/generator.hpp>

int main(int argc, const char *argv[]) {
    if (argc > 3) {
        std::ofstream dst(argv[2]);
        gridtools::c_bindings::generate_fortran_interface(dst, argv[3]);
    }
    if (argc > 1) {
        std::ofstream dst(argv[1]);
        gridtools::c_bindings::generate_c_interface(dst);
    } else {
        gridtools::c_bindings::generate_c_interface(std::cout);
    }
}

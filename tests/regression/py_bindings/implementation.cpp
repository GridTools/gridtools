/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <utility>

#include <pybind11/pybind11.h>

#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/stencil/global_parameter.hpp>
#include <gridtools/stencil/naive.hpp>
#include <gridtools/storage/adapter/python_sid_adapter.hpp>

namespace py = pybind11;

using namespace gridtools;
using namespace stencil;
using namespace cartesian;

struct copy_functor {
    using in = in_accessor<0>;
    using out = inout_accessor<1>;
    using param_list = make_param_list<in, out>;
    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        eval(out()) = eval(in());
    }
};

// Here is the generic implementation of the copy algorithm.
// The input and output fields are passed as SIDs
template <class From, class To>
void copy(From &&from, To &&to) {
    static_assert(is_sid<From>(), "");
    static_assert(is_sid<To>(), "");
    auto &&size = sid::get_upper_bounds(to);
    run_single_stage(copy_functor(),
        naive(),
        make_grid(at_key<dim::i>(size), at_key<dim::j>(size), at_key<dim::k>(size)),
        std::forward<From>(from),
        std::forward<To>(to));
}

// The module exports several instantiations of the generic `copy` to python.
// The differences between exported functions are in the way how parameters model the SID concept.
// Note that the generic algorithm stays the same.
PYBIND11_MODULE(py_implementation, m) {
    m.def("copy_from_3D",
        [](py::buffer from, py::buffer to) { copy(as_sid<double const, 3>(from), as_sid<double, 3>(to)); },
        "Copy from one buffer 3D buffer of doubles to another.");
    m.def("copy_from_1D",
        [](py::buffer from, py::buffer to) { copy(as_sid<double const, 1>(from), as_sid<double, 3>(to)); },
        "Copy from the 1D double buffer to a 3D one.");
    m.def("copy_from_scalar",
        [](double from, py::buffer to) { copy(make_global_parameter(from), as_sid<double, 3>(to)); },
        "Copy from the 1D double buffer to a 3D one.");
}

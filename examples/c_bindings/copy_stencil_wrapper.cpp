/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// In this example, we demonstrate how the cpp_bindgen library can be used to export functions to C and Fortran. We are
// going to export the functions required to run a simple copy stencil (see also the commented example in
// examples/stencil_composition/copy_stencil.cpp)

#include <cassert>
#include <functional>

#include <cpp_bindgen/export.hpp>

#include <gridtools/interface/fortran_array_adapter.hpp>
#include <gridtools/stencil_composition/cartesian.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#ifdef __CUDACC__
#include <gridtools/stencil_composition/backend/cuda.hpp>
#include <gridtools/storage/cuda.hpp>
using backend_t = gridtools::cuda::backend<>;
using storage_traits_t = gridtools::storage::cuda;
#else
#include <gridtools/stencil_composition/backend/mc.hpp>
#include <gridtools/storage/mc.hpp>
using backend_t = gridtools::mc::backend;
using storage_traits_t = gridtools::storage::mc;
#endif

namespace {
    using namespace gridtools;
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

    auto make_data_store_impl(int x, int y, int z) {
        return storage::builder<storage_traits_t>.type<float>().dimensions(x, y, z).build();
    }
    BINDGEN_EXPORT_BINDING_3(make_data_store, make_data_store_impl);

    using data_store_ptr_t = decltype(make_data_store_impl(0, 0, 0));
    using data_store_t = typename data_store_ptr_t::element_type;

    void run_copy_stencil_impl(data_store_ptr_t in, data_store_ptr_t out) {
        assert(in->lengths() == out->lengths());
        auto &&lengths = out->lengths();
        auto grid = make_grid(lengths[0], lengths[1], lengths[2]);
        easy_run(copy_functor(), backend_t(), grid, in, out);
#ifdef __CUDACC__
        cudaDeviceSynchronize();
#endif
    }
    BINDGEN_EXPORT_BINDING_2(run_copy_stencil, run_copy_stencil_impl);

    // Note that fortran_array_adapters are "fortran array wrappable".
    static_assert(c_bindings::is_fortran_array_wrappable<fortran_array_adapter<data_store_t>>::value, "");

    void transform_f_to_c_impl(data_store_ptr_t data_store, fortran_array_adapter<data_store_t> descriptor) {
        transform(data_store, descriptor);
    }
    // In order to generate the additional wrapper for Fortran array, the *_WRAPPED_* versions need to be used
    BINDGEN_EXPORT_BINDING_WRAPPED_2(transform_f_to_c, transform_f_to_c_impl);

    void transform_c_to_f_impl(fortran_array_adapter<data_store_t> descriptor, data_store_ptr_t data_store) {
        transform(descriptor, data_store);
    }
    // In order to generate the additional wrapper for Fortran array, the *_WRAPPED_* versions need to be used
    BINDGEN_EXPORT_BINDING_WRAPPED_2(transform_c_to_f, transform_c_to_f_impl);
} // namespace

/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cpp_bindgen/export.hpp>
#include <gridtools/interface/fortran_array_adapter.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>

namespace gt = gridtools;

// In this example, we demonstrate how the cpp_bindgen library can be used to export functions to C and Fortran. We are
// going to export the functions required to run a simple copy stencil (see also the commented example in
// examples/stencil_composition/copy_stencil.cpp)

namespace {

    using axis_t = gt::axis<1>::axis_interval_t;
    using grid_t = gt::grid<axis_t>;

#ifdef __CUDACC__
    using backend_t = gt::backend::cuda;
#else
    using backend_t = gt::backend::mc;
#endif

    using storage_traits_t = gt::storage_traits<backend_t>;
    using storage_info_t = storage_traits_t::storage_info_t<0, 3>;
    using data_store_t = storage_traits_t::data_store_t<float, storage_info_t>;

    struct copy_functor {
        using in = gt::in_accessor<0>;
        using out = gt::inout_accessor<1>;
        using param_list = gt::make_param_list<in, out>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out{}) = eval(in{});
        }
    };

    using p_in = gt::arg<0, data_store_t>;
    using p_out = gt::arg<1, data_store_t>;

    // The following are wrapper functions which will be exported to C/Fortran
    grid_t make_grid_impl(int nx, int ny, int nz) { return {gt::make_grid(nx, ny, nz)}; }
    storage_info_t make_storage_info_impl(int nx, int ny, int nz) { return {nx, ny, nz}; }
    data_store_t make_data_store_impl(storage_info_t storage_info) { return {storage_info}; }

    gt::computation<p_in, p_out> make_copy_stencil_impl(const grid_t &grid) {
        return gt::make_computation<backend_t>(
            grid, gt::make_multistage(gt::execute::parallel(), gt::make_stage<copy_functor>(p_in{}, p_out{})));
    }

    // Note that fortran_array_adapters are "fortran array wrappable".
    static_assert(gt::c_bindings::is_fortran_array_wrappable<gt::fortran_array_adapter<data_store_t>>::value, "");

    void transform_f_to_c_impl(data_store_t data_store, gt::fortran_array_adapter<data_store_t> descriptor) {
        transform(data_store, descriptor);
    }
    void transform_c_to_f_impl(gt::fortran_array_adapter<data_store_t> descriptor, data_store_t data_store) {
        transform(descriptor, data_store);
    }

    // That means that in the generated Fortran code, a wrapper is created that takes a Fortran array, and converts
    // it into the fortran array wrappable type.
    void run_stencil_impl(gt::computation<p_in, p_out> &computation, data_store_t in, data_store_t out) {

        computation.run(p_in() = in, p_out() = out);

#ifdef __CUDACC__
        cudaDeviceSynchronize();
#endif
    }

    // exports `make_grid_impl` (which needs 3 arguments) under the name `make_grid`
    BINDGEN_EXPORT_BINDING_3(make_grid, make_grid_impl);
    BINDGEN_EXPORT_BINDING_3(make_storage_info, make_storage_info_impl);
    BINDGEN_EXPORT_BINDING_1(make_data_store, make_data_store_impl);
    BINDGEN_EXPORT_BINDING_1(make_copy_stencil, make_copy_stencil_impl);
    BINDGEN_EXPORT_BINDING_3(run_stencil, run_stencil_impl);

    // In order to generate the additional wrapper for Fortran array,
    // the *_WRAPPED_* versions need to be used
    BINDGEN_EXPORT_BINDING_WRAPPED_2(transform_c_to_f, transform_c_to_f_impl);
    BINDGEN_EXPORT_BINDING_WRAPPED_2(transform_f_to_c, transform_f_to_c_impl);
} // namespace

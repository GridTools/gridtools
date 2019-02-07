/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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

#include <gridtools/c_bindings/export.hpp>
#include <gridtools/interface/fortran_array_adapter.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>

namespace gt = gridtools;

namespace {

    using axis_t = gt::axis<1>::axis_interval_t;
    using grid_t = gt::grid<axis_t>;

    using target_t = gt::target::mc;
    using strategy_t = gt::strategy::block;
    using backend_t = gt::backend<target_t, gt::grid_type::structured, strategy_t>;
    using storage_traits_t = gt::storage_traits<backend_t::backend_id_t>;
    using storage_info_t = storage_traits_t::storage_info_t<0, 3>;
    using data_store_t = storage_traits_t::data_store_t<float, storage_info_t>;

    struct copy_functor {
        using in = gt::accessor<0, gt::enumtype::in>;
        using out = gt::accessor<1, gt::enumtype::inout>;
        using arg_list = gt::make_arg_list<in, out>;

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(out{}) = eval(in{});
        }
    };

} // namespace

namespace {
    using p_in = gt::arg<0, data_store_t>;
    using p_out = gt::arg<1, data_store_t>;

    struct wrapper {
        data_store_t in;
        data_store_t out;
        grid_t grid;
    };

    wrapper make_wrapper_impl(int nx, int ny, int nz) {
        auto si = storage_info_t{nx, ny, nz};
        return {{si, "in_field"}, {si, "out_field"}, gt::make_grid(nx, ny, nz)};
    }

    gt::computation<p_in, p_out> make_copy_stencil_impl(const wrapper &wrapper) {
        return gt::make_computation<backend_t>(wrapper.grid,
            gt::make_multistage(
                gt::enumtype::execute<gt::enumtype::parallel>(), gt::make_stage<copy_functor>(p_in{}, p_out{})));
    }

    // Note that fortran_array_adapters are "fortran array wrappable".
    static_assert(gt::c_bindings::is_fortran_array_wrappable<gt::fortran_array_adapter<data_store_t>>::value, "");

    // That means that in the generated Fortran code, a wrapper is created that takes a Fortran array, and converts
    // it into the fortran array wrappable type.
    void run_stencil_impl(wrapper &wrapper,
        gt::computation<p_in, p_out> &computation,
        gt::fortran_array_adapter<data_store_t> in_f,
        gt::fortran_array_adapter<data_store_t> out_f) {

        transform(wrapper.in, in_f);

        computation.run(p_in() = wrapper.in, p_out() = wrapper.out);

        transform(out_f, wrapper.out);
    }

    GT_EXPORT_BINDING_3(make_wrapper, make_wrapper_impl);
    GT_EXPORT_BINDING_1(make_copy_stencil, make_copy_stencil_impl);

    // In order to generate the additional wrapper in Fortran, the *_WRAPPED_* versions need to be used
    GT_EXPORT_BINDING_WRAPPED_4(run_stencil, run_stencil_impl);
} // namespace

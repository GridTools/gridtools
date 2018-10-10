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

#include <functional>
#include <iostream>
#include <typeinfo>

#include <boost/mpl/vector.hpp>

#include <gridtools/c_bindings/export.hpp>

#include <gridtools/stencil-composition/stencil-composition.hpp>

#include "backend_select.hpp"

namespace {

    using namespace gridtools;
    using namespace enumtype;
    namespace m = boost::mpl;

    struct copy_functor {
        using in = accessor<0, enumtype::in, extent<>, 3>;
        using out = accessor<1, enumtype::inout, extent<>, 3>;
        using arg_list = m::vector<in, out>;

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(out{}) = eval(in{});
        }
    };

    using storage_info_t = storage_traits<backend_t::backend_id_t>::storage_info_t<0, 3>;

    template <class T>
    using generic_data_store_t = storage_traits<backend_t::backend_id_t>::data_store_t<T, storage_info_t>;

    using data_store_t = generic_data_store_t<float_type>;
} // namespace

namespace gridtools {
    template <typename T, typename = enable_if_t<is_data_store<remove_const_t<T>>::value>>
    T gt_make_fortran_array_view(gt_fortran_array_descriptor *descriptor, T *) {
        if (descriptor->rank != 3) {
            throw std::runtime_error("only 3-dimensional arrays are supported");
        }
        return T(storage_info_t(descriptor->dims[0], descriptor->dims[1], descriptor->dims[2]),
            reinterpret_cast<typename T::data_t *>(descriptor->data));
    }
    template <typename T, typename = enable_if_t<is_data_store<remove_const_t<T>>::value>>
    gt_fortran_array_descriptor get_fortran_view_meta(T *) {
        gt_fortran_array_descriptor descriptor;
        descriptor.type = c_bindings::fortran_array_element_kind<typename T::data_t>::value;
        descriptor.rank = 3;
        descriptor.is_acc_present = false;
        return descriptor;
    }

    static_assert(c_bindings::is_fortran_array_bindable<generic_data_store_t<double>>::value, "");
    static_assert(c_bindings::is_fortran_array_wrappable<generic_data_store_t<double>>::value, "");
} // namespace gridtools
namespace {
    GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED_1(sync_data_store, void(data_store_t), std::mem_fn(&data_store_t::sync));

    using p_in = arg<0, data_store_t>;
    using p_out = arg<1, data_store_t>;

    auto make_grid(data_store_t data_store) -> decltype(make_grid(0, 0, 0)) {
        auto dims = data_store.total_lengths();
        return gridtools::make_grid(dims[0], dims[1], dims[2]);
    }

    auto make_copy_stencil(data_store_t in, data_store_t out) GT_AUTO_RETURN(make_computation<backend_t>(make_grid(out),
        p_in{} = in,
        p_out{} = out,
        make_multistage(execute<forward>(), make_stage<copy_functor>(p_in{}, p_out{}))));
    GT_EXPORT_BINDING_WRAPPED_2(create_copy_stencil, make_copy_stencil);

    using stencil_t = decltype(make_copy_stencil(std::declval<data_store_t>(), std::declval<data_store_t>()));

    GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED_1(run_stencil, void(stencil_t &), std::mem_fn(&stencil_t::run<>));
} // namespace

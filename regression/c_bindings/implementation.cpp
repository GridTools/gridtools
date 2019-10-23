/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <functional>
#include <iostream>
#include <typeinfo>

#include <cpp_bindgen/export.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/tools/backend_select.hpp>

namespace {

    using namespace gridtools;

    struct copy_functor {
        using in = accessor<0, intent::in, extent<>, 3>;
        using out = accessor<1, intent::inout, extent<>, 3>;
        using param_list = make_param_list<in, out>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out{}) = eval(in{});
        }
    };

    using storage_info_t = storage_traits<backend_t>::storage_info_t<0, 3>;

    template <class T>
    using generic_data_store_t = storage_traits<backend_t>::data_store_t<T, storage_info_t>;

    using data_store_t = generic_data_store_t<float_type>;

    template <class T>
    generic_data_store_t<T> make_data_store(uint_t x, uint_t y, uint_t z, T *ptr) {
        return generic_data_store_t<T>(storage_info_t(x, y, z), ptr);
    }
    BINDGEN_EXPORT_GENERIC_BINDING(4, generic_create_data_store, make_data_store, (double)(float));
    BINDGEN_EXPORT_BINDING_4(create_data_store, make_data_store<float_type>);

    BINDGEN_EXPORT_BINDING_WITH_SIGNATURE_1(sync_data_store, void(data_store_t), std::mem_fn(&data_store_t::sync));

    auto make_grid(data_store_t data_store) {
        auto dims = data_store.total_lengths();
        return gridtools::make_grid(dims[0], dims[1], dims[2]);
    }

    auto make_copy_stencil(data_store_t const &in, data_store_t const &out) {
        return [=] { easy_run(copy_functor(), backend_t(), make_grid(out), in, out); };
    }
    BINDGEN_EXPORT_BINDING_2(create_copy_stencil, make_copy_stencil);

    using stencil_t = decltype(make_copy_stencil(std::declval<data_store_t>(), std::declval<data_store_t>()));

    BINDGEN_EXPORT_BINDING_WITH_SIGNATURE_1(run_stencil, void(stencil_t const &), [](auto &&f) { f(); });
} // namespace

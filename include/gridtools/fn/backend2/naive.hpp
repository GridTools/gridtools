/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../common/functional.hpp"
#include "../../common/hymap.hpp"
#include "../../sid/allocator.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/contiguous.hpp"
#include "./common.hpp"

namespace gridtools::fn::backend {
    namespace naive_impl_ {
        struct naive {};

        template <class Sizes, class StencilStage, class Composite>
        void apply_stencil_stage(naive, Sizes const &sizes, StencilStage, Composite composite) {
            auto ptr = sid::get_origin(composite)();
            auto strides = sid::get_strides(composite);
            common::make_loops(sizes)(StencilStage())(ptr, strides);
        }

        template <class Vertical, class Sizes, class ColumnStage, class Composite, class Seed>
        void apply_column_stage(naive, Sizes const &sizes, ColumnStage, Composite composite, Seed seed) {
            auto ptr = sid::get_origin(composite)();
            auto strides = sid::get_strides(composite);
            auto v_size = at_key<Vertical>(sizes);
            common::make_loops(hymap::canonicalize_and_remove_key<Vertical>(sizes))(
                [v_size = std::move(v_size), seed = std::move(seed)](auto ptr, auto const &strides) {
                    ColumnStage()(seed, v_size, std::move(ptr), strides);
                })(ptr, strides);
        }

        inline auto tmp_allocator(naive) { return sid::make_allocator(&std::make_unique<char[]>); }

        template <class T, class Allocator, class Sizes>
        auto allocate_global_tmp(naive, Allocator &allocator, Sizes const &sizes) {
            return sid::make_contiguous<T, int_t>(allocator, sizes);
        }
    } // namespace naive_impl_

    using naive_impl_::naive;

    using naive_impl_::apply_column_stage;
    using naive_impl_::apply_stencil_stage;

    using naive_impl_::allocate_global_tmp;
    using naive_impl_::tmp_allocator;
} // namespace gridtools::fn::backend

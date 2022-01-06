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
#include "../../meta/rename.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/loop.hpp"

namespace gridtools::fn::backend {
    namespace naive_impl_ {
        struct naive {};

        template <class Sizes>
        auto make_loops(Sizes const &sizes) {
            return tuple_util::fold(
                [&]<class Outer, class Dim>(Outer outer, Dim) {
                    return [ outer = std::move(outer), inner = sid::make_loop<Dim>(at_key<Dim>(sizes)) ]<class... Args>(
                        Args && ... args) {
                        return outer(inner(std::forward<Args>(args)...));
                    };
                },
                identity(),
                meta::rename<std::tuple, get_keys<Sizes>>());
        }

        template <class Sizes, class StencilStage, class Composite>
        void apply_stencil_stage(naive, Sizes const &sizes, StencilStage, Composite composite) {
            auto ptr = sid::get_origin(composite)();
            auto strides = sid::get_strides(composite);
            make_loops(sizes)(StencilStage())(ptr, strides);
        }

        template <class Vertical, class Sizes, class ColumnStage, class Composite, class Seed>
        void apply_column_stage(naive, Sizes const &sizes, ColumnStage, Composite composite, Seed seed) {
            auto ptr = sid::get_origin(composite)();
            auto strides = sid::get_strides(composite);
            auto v_size = at_key<Vertical>(sizes);
            make_loops(hymap::remove_key<Vertical>(sizes))(
                [v_size = std::move(v_size), seed = std::move(seed)](auto ptr, auto const &strides) {
                    ColumnStage()(seed, v_size, std::move(ptr), strides);
                })(ptr, strides);
        }
    } // namespace naive_impl_

    using naive_impl_::apply_column_stage;
    using naive_impl_::apply_stencil_stage;
    using naive_impl_::naive;
} // namespace gridtools::fn::backend

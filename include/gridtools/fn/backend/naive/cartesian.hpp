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

#include <tuple>
#include <utility>

#include "../../../common/tuple_util.hpp"
#include "../../../meta.hpp"
#include "../../../sid/composite.hpp"
#include "../../ast/vertical.hpp"
#include "../../cartesian.hpp"
#include "../../strided_iter.hpp"
#include "apply.hpp"
#include "tag.hpp"

namespace gridtools::fn {
    namespace cartesian_naive_impl_ {
        constexpr auto make_composite = []<class OutTag, class InTags, class Out, class Ins>(
                                            OutTag, InTags, Out &out, Ins &&ins) {
            using keys_t = meta::rename<sid::composite::keys, meta::push_back<InTags, OutTag>>;
            return tuple_util::convert_to<keys_t::template values>(tuple_util::push_back(std::forward<Ins>(ins), out));
        };
        constexpr auto make_iterator = [](auto &&ptr, auto const &strides) {
            return [&](auto tag) { return strided_iter(tag, ptr, strides); };
        };

    } // namespace cartesian_naive_impl_

    template <auto Stencil, class Sizes, class Offsets, class Output, class Inputs>
    void fn_apply(cartesian<Sizes, Offsets> const &domain, meta::val<Stencil>, Output &output, Inputs inputs) {
        using namespace cartesian_naive_impl_;
        naive_apply<Stencil>(domain.sizes, domain.offsets, output, std::move(inputs), make_composite, make_iterator);
    }

    template <class ScanTag, class Sizes, class Offsets, class Vertical, class Output, class Inputs>
    void fn_apply_scan(cartesian<Sizes, Offsets, Vertical> const &domain, ScanTag, Output &output, Inputs inputs) {
        using namespace cartesian_naive_impl_;
        naive_apply_scan<Vertical>(
            ScanTag(), domain.sizes, domain.offsets, output, std::move(inputs), make_composite, make_iterator);
    }
} // namespace gridtools::fn

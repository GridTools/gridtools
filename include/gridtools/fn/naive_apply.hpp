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
#include <type_traits>
#include <utility>

#include "../common/hymap.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "../sid/concept.hpp"
#include "../sid/loop.hpp"
#include "../sid/multi_shift.hpp"

namespace gridtools::fn {
    namespace naive_apply_impl_ {
        struct out_tag {};

        template <class T>
        struct in_tag : T {};

        template <class Sizes>
        auto make_loops(Sizes const &sizes) {
            return tuple_util::fold(
                [&]<class Outer, class Dim>(Outer outer, Dim) {
                    return [ outer = std::move(outer), inner = sid::make_loop<Dim>(at_key<Dim>(sizes)) ]<class... Args>(
                        Args && ... args) {
                        return outer(inner(std::forward<Args>(args)...));
                    };
                },
                std::identity(),
                meta::rename<std::tuple, get_keys<Sizes>>());
        }

        template <auto Stencil,
            class Sizes,
            class Offsets,
            class Output,
            class Inputs,
            class MakeComposite,
            class MakeIterator>
        void naive_apply(Sizes const &sizes,
            Offsets const &offsets,
            Output &output,
            Inputs &&inputs,
            MakeComposite &&make_composite,
            MakeIterator &&make_iterator) {
            using in_tags_t = meta::transform<in_tag, meta::make_indices<tuple_util::size<Inputs>, std::tuple>>;
            auto composite = std::forward<MakeComposite>(make_composite)(out_tag(), in_tags_t(), output, inputs);
            auto strides = sid::get_strides(composite);
            auto ptr = sid::get_origin(composite)();
            sid::multi_shift(ptr, strides, offsets);
            make_loops(sizes)([&](auto const &ptr, auto const &strides) {
                *at_key<out_tag>(ptr) =
                    std::apply(Stencil, tuple_util::transform(make_iterator(ptr, strides), in_tags_t()));
            })(ptr, strides);
        }
    } // namespace naive_apply_impl_
    using naive_apply_impl_::naive_apply;
} // namespace gridtools::fn

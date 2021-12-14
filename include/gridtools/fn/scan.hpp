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

#include <cassert>
#include <type_traits>

#include "../common/functional.hpp"
#include "../common/integral_constant.hpp"
#include "../common/tuple.hpp"
#include "../sid/concept.hpp"

namespace gridtools::fn {
    namespace scan_impl_ {
        template <class F, class Projector = host_device::identity>
        struct scan_pass {
            F m_f;
            Projector m_p;
            constexpr scan_pass(F f, Projector p = {}) : m_f(f), m_p(p) {}
        };

        template <class T>
        struct is_scan_pass : std::false_type {};

        template <class F, class P>
        struct is_scan_pass<scan_pass<F, P>> : std::true_type {};

        template <bool IsBackward>
        struct base : std::bool_constant<IsBackward> {
            static GT_FUNCTION consteval auto prologue() { return tuple(); }
            static GT_FUNCTION consteval auto epilogue() { return tuple(); }
        };

        using fwd = base<false>;
        using bwd = base<true>;

        template <class Vertical, class ScanOrFold, class MakeIterator, int Out, int... Ins>
        struct column_stage {
            GT_FUNCTION auto operator()(auto seed, std::size_t size, auto ptr, auto const &strides) const {
                constexpr std::size_t prologue_size = std::tuple_size_v<decltype(ScanOrFold::prologue())>;
                constexpr std::size_t epilogue_size = std::tuple_size_v<decltype(ScanOrFold::epilogue())>;
                assert(size >= prologue_size + epilogue_size);
                using step_t = integral_constant<int, ScanOrFold::value ? -1 : 1>;
                auto const &v_stride = sid::get_stride<Vertical>(strides);
                auto inc = [&] { sid::shift(ptr, v_stride, step_t()); };
                auto next = [&](auto acc, auto pass) {
                    if constexpr (is_scan_pass<decltype(pass)>()) {
                        // scan
                        auto res = pass.m_f(acc, MakeIterator()()(integral_constant<int, Ins>(), ptr, strides)...);
                        *at_key<integral_constant<int, Out>>(ptr) = pass.m_p(res);
                        inc();
                        return res;
                    } else {
                        // fold
                        auto res = pass(acc, MakeIterator()()(integral_constant<int, Ins>(), ptr, strides)...);
                        inc();
                        return res;
                    }
                };
                auto acc = tuple_util::fold(next, std::move(seed), ScanOrFold::prologue());
                std::size_t n = size - prologue_size - epilogue_size;
                for (std::size_t i = 0; i < n; ++i)
                    acc = next(std::move(acc), ScanOrFold::body());
                acc = tuple_util::fold(next, std::move(acc), ScanOrFold::epilogue());
                return acc;
            }
        };
    } // namespace scan_impl_

    using scan_impl_::bwd;
    using scan_impl_::column_stage;
    using scan_impl_::fwd;
    using scan_impl_::scan_pass;
} // namespace gridtools::fn

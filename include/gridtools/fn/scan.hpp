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

#include "../common/functional.hpp"
#include "../common/tuple.hpp"

namespace gridtools::fn {
    namespace scan_impl_ {
        template <class F, class Projector = host_device::identity>
        struct scan_pass {
            F m_f;
            Projector m_p;
            constexpr scan_pass(F f, Projector p = {}) : m_f(f), m_p(p) {}
        };

        template <bool IsBackward>
        struct base : std::bool_constant<IsBackward> {
            static GT_FUNCTION consteval auto prologue() { return tuple(); }
            static GT_FUNCTION consteval auto epilogue() { return tuple(); }
        };

        using fwd = base<false>;
        using bwd = base<true>;
    } // namespace scan_impl_

    using scan_impl_::bwd;
    using scan_impl_::fwd;
    using scan_impl_::scan_pass;
} // namespace gridtools::fn

/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 *   @file
 *
 *   Stage concept represents elementary functor from the backend implementor point of view.
 *
 *   Stage must have the nested `extent_t` type or an alias that has to model Extent concept.
 *   The meaning: the stage should be computed in the area that is extended from the user provided computation area by
 *   that much.
 *
 *   Stage also have static `exec` method that accepts an object by reference that models IteratorDomain.
 *   `exec` should execute an elementary functor in the grid point that IteratorDomain points to.
 *
 *   Note that the Stage is (and should stay) backend independent. The core of gridtools passes stages [split by k-loop
 *   intervals and independent groups] to the backend in the form of compile time only parameters.
 *
 *   TODO(anstaf): add `is_stage<T>` trait
 */

#pragma once

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../../meta.hpp"
#include "../accessor_intent.hpp"
#include "../arg.hpp"
#include "../esf_fwd.hpp"
#include "../expressions/expr_base.hpp"
#include "../has_apply.hpp"
#include "../positional.hpp"
#include "../sid/multi_shift.hpp"
#include "dim.hpp"
#include "extent.hpp"

namespace gridtools {
    namespace stage_impl_ {
        struct default_deref_f {
            template <class Arg, class T>
            GT_FUNCTION decltype(auto) operator()(T ptr) const {
                return *ptr;
            }
        };

        template <class Ptr, class Strides, class Args, class Deref>
        struct evaluator {
            Ptr const &GT_RESTRICT m_ptr;
            Strides const &GT_RESTRICT m_strides;

            template <class Arg>
            using ref_type =
                decltype(Deref{}.template operator()<Arg>(host_device::at_key<Arg>(std::declval<Ptr const &>())));

            template <class Accessor, class Arg = meta::at_c<Args, Accessor::index_t::value>>
            GT_FUNCTION apply_intent_t<Accessor::intent_v, ref_type<Arg>> operator()(Accessor acc) const {
                auto ptr = host_device::at_key<Arg>(m_ptr);
                sid::multi_shift<Arg>(ptr, m_strides, wstd::move(acc));
                return Deref{}.template operator()<Arg>(ptr);
            }

            template <class Op, class... Ts>
            GT_FUNCTION auto operator()(expr<Op, Ts...> arg) const {
                return expressions::evaluation::value(*this, wstd::move(arg));
            }

            GT_FUNCTION int_t i() const { return *host_device::at_key<positional<dim::i>>(m_ptr); }
            GT_FUNCTION int_t j() const { return *host_device::at_key<positional<dim::j>>(m_ptr); }
            GT_FUNCTION int_t k() const { return *host_device::at_key<positional<dim::k>>(m_ptr); }
        };
    } // namespace stage_impl_

    /**
     *   A stage that is associated with an elementary functor.
     */
    template <class Functor, class Extent, class Esf>
    struct stage {
        GT_STATIC_ASSERT(has_apply<Functor>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_esf_descriptor<Esf>::value, GT_INTERNAL_ERROR);

        using esf_t = Esf;
        using args_t = typename Esf::args_t;
        using extent_t = Extent;

        static GT_FUNCTION Extent extent() { return {}; }

        template <class Deref = stage_impl_::default_deref_f, class Ptr, class Strides>
        GT_FUNCTION void operator()(Ptr const &GT_RESTRICT ptr, Strides const &GT_RESTRICT strides) const {
            stage_impl_::evaluator<Ptr, Strides, args_t, Deref> eval{ptr, strides};
            Functor::apply(eval);
        }
    };
} // namespace gridtools

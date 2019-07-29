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
            template <class T>
            GT_FUNCTION decltype(auto) operator()(T ptr) const {
                return *ptr;
            }
        };

        template <class Ptr, class Strides, class Keys, class Deref>
        struct evaluator {
            Ptr const &GT_RESTRICT m_ptr;
            Strides const &GT_RESTRICT m_strides;

            template <class Accessor>
            GT_FUNCTION decltype(auto) operator()(Accessor acc) const {
                using key_t = meta::at_c<Keys, Accessor::index_t::value>;
                auto ptr = host_device::at_key<key_t>(m_ptr);
                sid::multi_shift<key_t>(ptr, m_strides, wstd::move(acc));
                return apply_intent<Accessor::intent_v>(Deref()(ptr));
            }

            template <class Op, class... Ts>
            GT_FUNCTION auto operator()(expr<Op, Ts...> arg) const {
                return expressions::evaluation::value(*this, wstd::move(arg));
            }

            GT_FUNCTION int_t i() const { return *host_device::at_key<positional<dim::i>>(m_ptr); }
            GT_FUNCTION int_t j() const { return *host_device::at_key<positional<dim::j>>(m_ptr); }
            GT_FUNCTION int_t k() const { return *host_device::at_key<positional<dim::k>>(m_ptr); }
        };

        template <class Functor, class PlhMap>
        struct stage {
            GT_STATIC_ASSERT(has_apply<Functor>::value, GT_INTERNAL_ERROR);

            template <class Deref = void, class Ptr, class Strides>
            GT_FUNCTION void operator()(Ptr const &GT_RESTRICT ptr, Strides const &GT_RESTRICT strides) const {
                using deref_t = meta::if_<std::is_void<Deref>, default_deref_f, Deref>;
                using eval_t = evaluator<Ptr, Strides, PlhMap, deref_t>;
                eval_t eval{ptr, strides};
                Functor::template apply<eval_t &>(eval);
            }
        };
    } // namespace stage_impl_
    using stage_impl_::stage;
} // namespace gridtools

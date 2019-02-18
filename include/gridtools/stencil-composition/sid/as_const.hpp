/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>
#include <utility>

#include "../../meta/macros.hpp"
#include "../../meta/type_traits.hpp"
#include "concept.hpp"
#include "delegate.hpp"

namespace gridtools {
    namespace sid {
        namespace as_const_impl_ {
            template <class Sid>
            class const_adapter : public delegate<Sid> {
                friend GT_META_CALL(sid::element_type, Sid) const *sid_get_origin(const_adapter const &obj) {
                    return sid::get_origin(const_cast<Sid &>(obj.impl()));
                }
                using sid::delegate<Sid>::delegate;
            };
        } // namespace as_const_impl_

        /**
         *   Returns a `SID`, which ptr_type is a pointer to const.
         *   enabled only if the original ptr_type is a pointer.
         */
        template <class SrcRef,
            class Src = decay_t<SrcRef>,
            enable_if_t<std::is_pointer<GT_META_CALL(sid::ptr_type, Src)>::value, int> = 0>
        as_const_impl_::const_adapter<Src> as_const(SrcRef &&src) {
            return as_const_impl_::const_adapter<Src>{std::forward<SrcRef>(src)};
        }
    } // namespace sid
} // namespace gridtools

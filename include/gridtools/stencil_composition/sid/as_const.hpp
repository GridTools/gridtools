/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>
#include <utility>

#include "../../common/host_device.hpp"
#include "../../meta/macros.hpp"
#include "../../meta/type_traits.hpp"
#include "concept.hpp"
#include "delegate.hpp"
#include "simple_ptr_holder.hpp"

namespace gridtools {
    namespace sid {
        namespace as_const_impl_ {
            template <class Sid>
            class const_adapter : public delegate<Sid> {
                struct const_ptr_holder {
                    GT_META_CALL(sid::ptr_holder_type, Sid) m_impl;

                    GT_FUNCTION GT_META_CALL(sid::element_type, Sid) const *operator()() const { return m_impl(); }

                    friend constexpr const_ptr_holder operator+(const_ptr_holder const &obj, ptrdiff_t offset) {
                        return {obj.m_impl + offset};
                    }
                };

                friend const_ptr_holder sid_get_origin(const_adapter const &obj) {
                    return {sid::get_origin(const_cast<Sid &>(obj.impl()))};
                }
                using sid::delegate<Sid>::delegate;
            };
        } // namespace as_const_impl_

        /**
         *   Returns a `SID`, which ptr_type is a pointer to const.
         *   enabled only if the original ptr_type is a pointer.
         *
         *   TODO(anstaf): at a moment the generated ptr holder always has `host_device` `operator()`
         *                 probably might we need the `host` and `device` variations as well
         */
        template <class SrcRef,
            class Src = decay_t<SrcRef>,
            enable_if_t<std::is_pointer<GT_META_CALL(sid::ptr_type, Src)>::value, int> = 0>
        as_const_impl_::const_adapter<Src> as_const(SrcRef &&src) {
            return as_const_impl_::const_adapter<Src>{std::forward<SrcRef>(src)};
        }
    } // namespace sid
} // namespace gridtools

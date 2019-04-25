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

#include "../../common/generic_metafunctions/for_each.hpp"
#include "delegate.hpp"
#include "multi_shift.hpp"

namespace gridtools {
    namespace sid {
        namespace sid_helpers_impl_ {
            template <class Sid, class Offset>
            class shifted_sid : public delegate<Sid> {
                Offset m_offset;

                friend GT_META_CALL(sid::ptr_holder_type, Sid) sid_get_origin(shifted_sid &obj) {
                    auto &&impl = obj.impl();
                    auto strides = sid::get_strides(impl);
                    GT_META_CALL(sid::ptr_diff_type, Sid) offset{};
                    multi_shift(offset, strides, obj.m_offset);
                    return sid::get_origin(impl) + offset;
                }

              public:
                explicit constexpr shifted_sid(Sid const &impl, Offset const &offset) noexcept
                    : delegate<Sid>(impl), m_offset(offset) {}
            };
        } // namespace sid_helpers_impl_

        template <class Sid, class Offset>
        sid_helpers_impl_::shifted_sid<Sid, Offset> shifted_sid(Sid const &sid, Offset const &offset) {
            return sid_helpers_impl_::shifted_sid<Sid, Offset>{sid, offset};
        }
    } // namespace sid
} // namespace gridtools

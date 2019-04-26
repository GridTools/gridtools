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

#include "../../meta.hpp"
#include "concept.hpp"
#include "delegate.hpp"
#include "multi_shift.hpp"

namespace gridtools {
    namespace sid {
        namespace sid_helpers_impl_ {
            template <class Sid>
            class shifted_sid : public delegate<Sid> {
                GT_META_CALL(sid::ptr_holder_type, Sid) m_origin;

                friend GT_META_CALL(sid::ptr_holder_type, Sid) sid_get_origin(shifted_sid &obj) { return obj.m_origin; }

              public:
                template <class Offsets>
                shifted_sid(Sid const &original_sid, Offsets &&offsets) noexcept
                    : delegate<Sid>(original_sid), m_origin{[this, &offsets]() {
                          auto &&strides = sid::get_strides(this->impl());
                          GT_META_CALL(sid::ptr_diff_type, Sid) ptr_offset{};
                          multi_shift(ptr_offset, strides, offsets);
                          return sid::get_origin(this->impl()) + ptr_offset;
                      }()} {}
            };
        } // namespace sid_helpers_impl_

        template <class Sid, class Offset>
        sid_helpers_impl_::shifted_sid<decay_t<Sid>> shifted_sid(Sid &&sid, Offset &&offset) {
            return {std::forward<Sid>(sid), std::forward<Offset>(offset)};
        }
    } // namespace sid
} // namespace gridtools

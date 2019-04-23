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

#include "delegate.hpp"

namespace gridtools {
    namespace sid {
        namespace sid_helpers_impl_ {

            template <class Strides, class Offsets, class PtrDiff>
            struct offset_f {
                Strides const &strides;
                Offsets const &offsets;
                PtrDiff &ptr_diff;

                template <class Key>
                void operator()(Key) const {
                    sid::shift(ptr_diff, sid::get_stride<Key>(strides), integral_constant<int, 1>{});
                    //                    ptr_diff += at_key<T>(strides) * at_key<T>(offsets);
                }
            };

            //            TEST(for_each, functional) {
            //                double acc = 0;
            //                for_each(accumulate_f{acc}, std::make_tuple(42, 5.3));
            //                EXPECT_EQ(47.3, acc);
            //            }

            template <class Sid, class Offset>
            class shifted_sid : public delegate<Sid> {
                Offset m_offset;

                friend GT_META_CALL(sid::ptr_holder_type, Sid) sid_get_origin(shifted_sid &obj) {
                    auto &&impl = obj.impl();
                    GT_META_CALL(sid::ptr_diff_type, Sid) offset{};
                    sid::shift(offset,
                        sid::get_stride<integral_constant<int, 1>>(sid::get_strides(impl)),
                        integral_constant<int, 1>{});
                    sid::shift(offset,
                        sid::get_stride<integral_constant<int, 2>>(sid::get_strides(impl)),
                        integral_constant<int, 2>{});
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

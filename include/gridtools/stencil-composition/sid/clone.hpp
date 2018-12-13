/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../../meta/macros.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {
        namespace clone_impl_ {
            template <class Ptr,
                class Strides,
                class BoundsValidator,
                class PtrDiff,
                class StridesKind,
                class BoundsValidatorKind>
            struct cloned {
                Ptr m_origin;
                Strides m_strides;
                BoundsValidator m_bounds_validator;

                friend constexpr GT_FUNCTION Ptr sid_get_origin(cloned &obj) { return obj.m_origin; }
                friend constexpr GT_FUNCTION Strides sid_get_strides(cloned const &obj) { return obj.m_strides; }
                friend constexpr GT_FUNCTION BoundsValidator sid_get_bounds_validator(cloned const &obj) {
                    return obj.m_bounds_validator;
                }

                friend PtrDiff sid_get_ptr_diff(cloned const &) { return {}; }
            };

            template <class Ptr,
                class Strides,
                class BoundsValidator,
                class PtrDiff,
                class StridesKind,
                class BoundsValidatorKind>
            StridesKind sid_get_strides_kind(
                cloned<Ptr, Strides, BoundsValidator, PtrDiff, StridesKind, BoundsValidatorKind> const &);

            template <class Ptr,
                class Strides,
                class BoundsValidator,
                class PtrDiff,
                class StridesKind,
                class BoundsValidatorKind>
            BoundsValidatorKind sid_get_bounds_validator_kind(
                cloned<Ptr, Strides, BoundsValidator, PtrDiff, StridesKind, BoundsValidatorKind> const &);
        } // namespace clone_impl_

        template <class Sid,
            class Res = clone_impl_::cloned<GT_META_CALL(ptr_type, Sid),
                GT_META_CALL(strides_type, Sid),
                GT_META_CALL(bounds_validator_type, Sid),
                GT_META_CALL(ptr_diff_type, Sid),
                GT_META_CALL(strides_kind, Sid),
                GT_META_CALL(bounds_validator_kind, Sid)>>
        constexpr GT_FUNCTION Res clone(Sid &src) noexcept {
            GRIDTOOLS_STATIC_ASSERT(is_sid<Sid>::value, GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT(is_sid<Res>::value, GT_INTERNAL_ERROR);
            return {get_origin(src), get_strides(src), get_bounds_validator(src)};
        }
    } // namespace sid
} // namespace gridtools

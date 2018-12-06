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
#include "../../common/generic_metafunctions/utility.hpp"
#include "../../common/host_device.hpp"
#include "../../meta/macros.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {
        template <class Sid>
        class delegate {
          protected:
            GRIDTOOLS_STATIC_ASSERT(is_sid<Sid>::value, GT_INTERNAL_ERROR);

            Sid m_impl;

            explicit constexpr GT_FUNCTION delegate(Sid const &impl) noexcept : m_impl(impl) {}
            explicit constexpr GT_FUNCTION delegate(Sid &&impl) noexcept : m_impl(const_expr::move(impl)) {}

            friend constexpr GT_FUNCTION Ptr sid_get_origin(delegate &obj) { return sid::get_origin(obj.m_impl); }
            friend constexpr GT_FUNCTION Strides sid_get_strides(delegate const &obj) {
                return sid::get_strides(obj.m_impl);
            }
            friend constexpr GT_FUNCTION BoundsValidator sid_get_bounds_validator(delegate const &obj) {
                return sid::get_bounds_validator(obj.m_impl);
            }
            friend GT_META_CALL(sid::ptr_diff_type, Sid) PtrDiff sid_get_ptr_diff(delegate const &);
            friend GT_META_CALL(sid::strides_kind, Sid) sid_get_strides_kind(delegate const &);
            friend GT_META_CALL(sid::bounds_validator_kind, Sid) sid_get_bounds_validator_kind(delegate const &);
        };
    } // namespace sid
} // namespace gridtools

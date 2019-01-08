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
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/host_device.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta/macros.hpp"
#include "../../meta/make_indices.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {
        namespace multi_shift_impl_ {
            template <class Ptr, class Strides, class Offsets>
            struct shift_f {
                Ptr &RESTRICT m_ptr;
                Strides const &RESTRICT m_strides;
                Offsets const &RESTRICT m_offsets;

                template <class I>
                GT_FUNCTION void operator()() const {
                    shift(m_ptr, get_stride<I::value>(m_strides), tuple_util::host_device::get<I::value>(m_offsets));
                }
            };
        } // namespace multi_shift_impl_

        /**
         *   A helper the invokes `sid::shift` in several dimensions.
         *   `offsets` should be a tuple-like of individual offsets.
         */
        template <class Ptr, class Strides, class Offsets>
        GT_FUNCTION void multi_shift(
            Ptr &RESTRICT ptr, Strides const &RESTRICT strides, Offsets const &RESTRICT offsets) {
            using indices_t = GT_META_CALL(meta::make_indices, tuple_util::size<Offsets>);
            host_device::for_each_type<indices_t>(
                multi_shift_impl_::shift_f<Ptr, Strides, Offsets>{ptr, strides, offsets});
        }
    } // namespace sid
} // namespace gridtools

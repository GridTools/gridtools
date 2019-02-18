/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
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
                Ptr &GT_RESTRICT m_ptr;
                Strides const &GT_RESTRICT m_strides;
                Offsets const &GT_RESTRICT m_offsets;

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
            Ptr &GT_RESTRICT ptr, Strides const &GT_RESTRICT strides, Offsets const &GT_RESTRICT offsets) {
            using indices_t = GT_META_CALL(meta::make_indices, tuple_util::size<Offsets>);
            host_device::for_each_type<indices_t>(
                multi_shift_impl_::shift_f<Ptr, Strides, Offsets>{ptr, strides, offsets});
        }
    } // namespace sid
} // namespace gridtools

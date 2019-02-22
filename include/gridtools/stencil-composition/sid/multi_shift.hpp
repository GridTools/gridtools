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

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/host_device.hpp"
#include "../../common/hymap.hpp"
#include "../../meta/macros.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {
        namespace multi_shift_impl_ {
            template <class Ptr, class Strides, class Offsets>
            struct shift_f {
                Ptr &GT_RESTRICT m_ptr;
                Strides const &GT_RESTRICT m_strides;
                Offsets const &GT_RESTRICT m_offsets;

                template <class Key>
                GT_FUNCTION void operator()() const {
                    shift(m_ptr, get_stride<Key>(m_strides), host_device::at_key<Key>(m_offsets));
                }
            };
        } // namespace multi_shift_impl_

        /**
         *   A helper the invokes `sid::shift` in several dimensions.
         *   `offsets` should be a hymap of individual offsets.
         */
        template <class Ptr, class Strides, class Offsets>
        GT_FUNCTION void multi_shift(
            Ptr &GT_RESTRICT ptr, Strides const &GT_RESTRICT strides, Offsets const &GT_RESTRICT offsets) {
            using keys_t = GT_META_CALL(get_keys, Offsets);
            host_device::for_each_type<keys_t>(
                multi_shift_impl_::shift_f<Ptr, Strides, Offsets>{ptr, strides, offsets});
        }
    } // namespace sid
} // namespace gridtools

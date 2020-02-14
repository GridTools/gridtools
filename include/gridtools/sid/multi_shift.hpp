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

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/for_each.hpp"
#include "../common/host_device.hpp"
#include "../common/hymap.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {
        namespace multi_shift_impl_ {
            template <class Ptr, class Strides, class Offsets>
            struct shift_f {
                Ptr &m_ptr;
                Strides const &m_strides;
                Offsets m_offsets;

                template <class Key>
                GT_FUNCTION void operator()() const {
                    shift(m_ptr, get_stride<Key>(m_strides), gridtools::host_device::at_key<Key>(m_offsets));
                }
            };

            template <class Arg, class Ptr, class Strides, class Offsets>
            struct composite_strides_shift_f {
                Ptr &m_ptr;
                Strides const &m_strides;
                Offsets m_offsets;

                template <class Key>
                GT_FUNCTION void operator()() const {
                    shift(
                        m_ptr, get_stride_element<Arg, Key>(m_strides), gridtools::host_device::at_key<Key>(m_offsets));
                }
            };
        } // namespace multi_shift_impl_

        /**
         *   A helper the invokes `sid::shift` in several dimensions.
         *   `offsets` should be a hymap of individual offsets.
         */
        template <class Ptr,
            class Strides,
            class Offsets,
            std::enable_if_t<tuple_util::size<Offsets>::value != 0, int> = 0>
        GT_FUNCTION void multi_shift(Ptr &ptr, Strides const &strides, Offsets offsets) {
            gridtools::host_device::for_each_type<get_keys<Offsets>>(
                multi_shift_impl_::shift_f<Ptr, Strides, Offsets>{ptr, strides, wstd::move(offsets)});
        }

        template <class Ptr,
            class Strides,
            class Offsets,
            std::enable_if_t<tuple_util::size<Offsets>::value == 0, int> = 0>
        GT_FUNCTION void multi_shift(Ptr &, Strides const &, Offsets) {}

        /**
         *   Variation of multi_shift that works with the strides of composite sid.
         */
        template <class Arg,
            class Ptr,
            class Strides,
            class Offsets,
            std::enable_if_t<tuple_util::size<Offsets>::value != 0, int> = 0>
        GT_FUNCTION void multi_shift(Ptr &ptr, Strides const &strides, Offsets offsets) {
            gridtools::host_device::for_each_type<get_keys<Offsets>>(
                multi_shift_impl_::composite_strides_shift_f<Arg, Ptr, Strides, Offsets>{
                    ptr, strides, wstd::move(offsets)});
        }

        template <class Arg,
            class Ptr,
            class Strides,
            class Offsets,
            std::enable_if_t<tuple_util::size<Offsets>::value == 0, int> = 0>
        GT_FUNCTION void multi_shift(Ptr &, Strides const &, Offsets) {}
    } // namespace sid
} // namespace gridtools

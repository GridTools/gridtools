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
#include "../common/for_each.hpp"
#include "../common/host_device.hpp"
#include "../common/hymap.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {
        /**
         *   A helper the invokes `sid::shift` in several dimensions.
         *   `offsets` should be a hymap of individual offsets.
         */
        template <class Ptr, class Strides, class Offsets>
        GT_FUNCTION void multi_shift(Ptr &ptr, Strides const &strides, Offsets offsets) {
            gridtools::host_device::for_each<meta::transform<meta::lazy::id, get_keys<Offsets>>>([&](auto key) {
                using key_t = typename decltype(key)::type;
                shift(ptr, get_stride<key_t>(strides), gridtools::host_device::at_key<key_t>(offsets));
            });
        }

        template <class Ptr, class Strides, class Offsets>
        GT_FUNCTION Ptr multi_shifted(Ptr ptr, Strides const &strides, Offsets offsets) {
            multi_shift(ptr, strides, wstd::move(offsets));
            return ptr;
        }

        /**
         *   Variation of multi_shift that works with the strides of composite sid.
         */
        template <class Arg, class Ptr, class Strides, class Offsets>
        GT_FUNCTION void multi_shift(Ptr &ptr, Strides const &strides, Offsets offsets) {
            gridtools::host_device::for_each<meta::transform<meta::lazy::id, get_keys<Offsets>>>([&](auto key) {
                using key_t = typename decltype(key)::type;
                shift(ptr, get_stride_element<Arg, key_t>(strides), gridtools::host_device::at_key<key_t>(offsets));
            });
        }

        template <class Arg, class Ptr, class Strides, class Offsets>
        GT_FUNCTION Ptr multi_shifted(Ptr ptr, Strides const &strides, Offsets offsets) {
            multi_shift<Arg>(ptr, strides, wstd::move(offsets));
            return ptr;
        }
    } // namespace sid
} // namespace gridtools

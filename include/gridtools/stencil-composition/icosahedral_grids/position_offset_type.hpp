/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>

#include "../../common/array.hpp"
#include "../../common/defs.hpp"

namespace gridtools {
    /**
     * @brief The position_offset is an array that keeps the iteration indices over a multidimensional domain.
     */
    using position_offset_type = array<int_t, 4>;

    template <size_t N>
    using position_offsets_type = array<position_offset_type, N>;

    template <class T>
    using is_position_offset_type = std::is_same<T, position_offset_type>;
} // namespace gridtools

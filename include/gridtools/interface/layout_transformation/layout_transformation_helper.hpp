/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "../../common/array.hpp"
#include "../../common/defs.hpp"
#include <vector>

namespace gridtools {
    namespace impl {
        /**
         * @brief copy std::vector to (potentially bigger) gridtools::array
         */
        template <size_t MaxDim>
        gridtools::array<gridtools::uint_t, MaxDim> vector_to_array(
            const std::vector<uint_t> &v, gridtools::uint_t init_value) {
            assert(MaxDim >= v.size() && "array too small");

            gridtools::array<gridtools::uint_t, MaxDim> a;
            std::fill(a.begin(), a.end(), init_value);
            std::copy(v.begin(), v.end(), a.begin());
            return a;
        }

        template <size_t MaxDim>
        gridtools::array<gridtools::uint_t, MaxDim> vector_to_dims_array(const std::vector<uint_t> &v) {
            return vector_to_array<MaxDim>(v, 1);
        }

        template <size_t MaxDim>
        gridtools::array<gridtools::uint_t, MaxDim> vector_to_strides_array(const std::vector<uint_t> &v) {
            return vector_to_array<MaxDim>(v, 0);
        }
    } // namespace impl
} // namespace gridtools

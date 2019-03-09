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

namespace gridtools {
    template <int_t I, ushort_t NColors>
    struct location_type {
        static const int_t value = I;
        typedef static_ushort<NColors> n_colors; //! <- is the number of locations of this type
    };

    template <typename T>
    struct is_location_type : std::false_type {};

    template <int I, ushort_t NColors>
    struct is_location_type<location_type<I, NColors>> : std::true_type {};

    namespace enumtype {
        typedef static_int<0> cells_index;
        typedef static_int<1> edges_index;
        typedef static_int<2> vertices_index;

        typedef location_type<cells_index::value, 2> cells;
        typedef location_type<edges_index::value, 3> edges;
        typedef location_type<vertices_index::value, 1> vertices;

        // a null or default location type indicate the absence of location type, therefore coloring is not applied,
        // and therefore the number of colors should be 1 (used in other parts of the code)
        typedef location_type<-1, 1> default_location_type;
    } // namespace enumtype

} // namespace gridtools

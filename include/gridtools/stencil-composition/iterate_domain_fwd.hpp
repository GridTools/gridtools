/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>

namespace gridtools {
    template <class>
    struct is_iterate_domain : std::false_type {};
} // namespace gridtools

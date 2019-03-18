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

#include "../meta/is_instantiation_of.hpp"
#include "../meta/macros.hpp"
#include "./backend_fwd.hpp"

namespace gridtools {
    template <class T>
    GT_META_DEFINE_ALIAS(is_backend, meta::is_instantiation_of, (backend, T));
} // namespace gridtools

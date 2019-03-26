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

#include "../common/defs.hpp"
#include "../meta/is_instantiation_of.hpp"
#include "../meta/macros.hpp"

namespace gridtools {

    /**
     * @brief metadata with the information for architecture
     * @tparam BackendId architecture backend id
     */
    template <class BackendId>
    struct backend_ids {
        using backend_id_t = BackendId;
    };

    template <class T>
    GT_META_DEFINE_ALIAS(is_backend_ids, meta::is_instantiation_of, (backend_ids, T));
} // namespace gridtools

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

#include "../common/gt_assert.hpp"
#include "../common/selector.hpp"
#include "./common/definitions.hpp"
#include "./common/storage_traits_metafunctions.hpp"
#include "./storage_cuda/cuda_storage.hpp"
#include "./storage_cuda/cuda_storage_info.hpp"

namespace gridtools {
    /** \ingroup storage
     * @{
     */

    template <class>
    struct storage_traits_from_id;

    /** @brief storage traits for the CUDA backend*/
    template <>
    struct storage_traits_from_id<backend::cuda> {
#ifdef __HIPCC__
        static constexpr uint_t default_alignment = 16;
#else
        static constexpr uint_t default_alignment = 32;
#endif

        template <typename ValueType>
        using select_storage = cuda_storage<ValueType>;

        template <uint_t Dims>
        using select_layout = typename get_layout<Dims, false>::type;
    };

    /**
     * @}
     */
} // namespace gridtools

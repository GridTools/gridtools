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
#include "../common/host_device.hpp"

/**
   @file
   @brief On all boundary the values ares set to DataField::value_type(), which is zero for basic data types.
*/

namespace gridtools {

    /** \ingroup Boundary-Conditions
     * @{
     */

    /**
       @brief On all boundary the values ares set to DataField::value_type(), which is zero for basic data types.
    */
    struct zero_boundary {

        template <typename Direction, typename DataField0>
        GT_FUNCTION void operator()(Direction, DataField0 &data_field0, uint_t i, uint_t j, uint_t k) const {
            data_field0(i, j, k) = typename DataField0::data_t();
        }

        template <typename Direction, typename DataField0, typename DataField1>
        GT_FUNCTION void operator()(
            Direction, DataField0 &data_field0, DataField1 &data_field1, uint_t i, uint_t j, uint_t k) const {
            data_field0(i, j, k) = typename DataField0::data_t();
            data_field1(i, j, k) = typename DataField1::data_t();
        }

        template <typename Direction, typename DataField0, typename DataField1, typename DataField2>
        GT_FUNCTION void operator()(Direction,
            DataField0 &data_field0,
            DataField1 &data_field1,
            DataField2 &data_field2,
            uint_t i,
            uint_t j,
            uint_t k) const {
            data_field0(i, j, k) = typename DataField0::data_t();
            data_field1(i, j, k) = typename DataField1::data_t();
            data_field2(i, j, k) = typename DataField2::data_t();
        }
    };

    /** @} */

} // namespace gridtools

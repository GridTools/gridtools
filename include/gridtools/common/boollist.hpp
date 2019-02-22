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

#include "../meta/type_traits.hpp"
#include "array.hpp"
#include "defs.hpp"
#include "host_device.hpp"

/*@file
@brief  The following class describes a boolean list of length N.

*/
namespace gridtools {

    /** \ingroup common
        @{
        \defgroup boollist List of Booleans
        @{
    */

    /**
       The following class describes a boolean list of length N.
       This is used in proc_grids.

       \code
       boollist<4> bl(true, false, false, true);
       bl.value3 == true
       bl.value2 == false
       \endcode
       See \link Concepts \endlink, \link proc_grid_2D_concept \endlink, \link proc_grid_3D_concept \endlink
     */
    template <ushort_t I>
    struct boollist {
        static const ushort_t m_size = I;

      private:
        // const
        array<bool, I> m_value;

      public:
        GT_FUNCTION
        constexpr ushort_t const &size() const { return m_size; }

        GT_FUNCTION
        constexpr bool const &value(ushort_t const &id) const { return m_value[id]; }
        GT_FUNCTION
        constexpr array<bool, I> const &value() const { return m_value; }

        GT_FUNCTION
        boollist(bool v0) : m_value{v0} {}

        GT_FUNCTION
        boollist(bool v0, bool v1) : m_value{v0, v1} {}

        GT_FUNCTION
        boollist(bool v0, bool v1, bool v2) : m_value{v0, v1, v2} {}

        GT_FUNCTION
        boollist(boollist const &bl) : m_value(bl.m_value) {}

        GT_FUNCTION
        void copy_out(bool *arr) const {
            for (ushort_t i = 0; i < I; ++i)
                arr[i] = m_value[i];
        }

        template <typename LayoutMap>
        GT_FUNCTION boollist<LayoutMap::masked_length> permute(
            enable_if_t<LayoutMap::masked_length == 1> *a = 0) const {
            return boollist<LayoutMap::masked_length>(m_value[LayoutMap::template find<0>()]);
        }

        template <typename LayoutMap>
        GT_FUNCTION boollist<LayoutMap::masked_length> permute(
            enable_if_t<LayoutMap::masked_length == 2> *a = 0) const {
            return boollist<LayoutMap::masked_length>(
                m_value[LayoutMap::template find<0>()], m_value[LayoutMap::template find<1>()]);
        }

        template <typename LayoutMap>
        GT_FUNCTION boollist<LayoutMap::masked_length> permute(
            enable_if_t<LayoutMap::masked_length == 3> *a = 0) const {
            return boollist<LayoutMap::masked_length>(m_value[LayoutMap::template find<0>()],
                m_value[LayoutMap::template find<1>()],
                m_value[LayoutMap::template find<2>()]);
        }
    };
    /** @} */
    /** @} */
} // namespace gridtools

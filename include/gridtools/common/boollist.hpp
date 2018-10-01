/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#include "array.hpp"
#include "defs.hpp"
#include "generic_metafunctions/type_traits.hpp"
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

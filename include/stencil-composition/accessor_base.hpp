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

#include <iostream>
#include <type_traits>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/type_traits/integral_constant.hpp>

#include "../gridtools.hpp"

#include "../storage/storage-facility.hpp"

#include "../common/array.hpp"
#include "../common/dimension.hpp"
#include "../common/functional.hpp"
#include "extent.hpp"
#include "arg_fwd.hpp"
#include "accessor_metafunctions.hpp"

namespace gridtools {

    /**
     * @brief Type to be used in elementary stencil functions to specify argument mapping and extents
     *
     One accessor consists substantially of an array of offsets (runtime values), a extent and an index (copmpile-time
     constants). The latter is used to distinguish the types of two different accessors,
     while the offsets are used to calculate the final memory address to be accessed by the stencil in \ref
     gridtools::iterate_domain.
     * The class also provides the interface for accessing data in the function body.
     The interfaces available to specify the offset in each dimension are covered in the following example, supposing
     that we have to specify the offsets of a 3D field V:
     - specify three offsets: in this case the order matters, the three arguments represent the offset in the  i, j, k
     directions respectively.
     \verbatim
     V(1,0,-3)
     \endverbatim
     - specify only some components: in this case the order is arbitrary and the missing components have offset zero;
     \verbatim
     V(z(-3),x(1))
     \endverbatim
     *
     * @tparam I Index of the argument in the function argument list
     * @tparam Extent Bounds over which the function access the argument
     */
    template < ushort_t Dim >
    class accessor_base {
        array< int_t, Dim > m_offsets;

        template < ushort_t Idx >
        int add_dimension(dimension< Idx > dim) {
            m_offsets[Idx - 1] += dim.value;
            return 0;
        }

        template < ushort_t Idx >
        using is_dimension_index = meta::bool_constant< (Idx > 0 && Idx <= Dim) >;

      public:
        static const ushort_t n_dimensions = Dim;
        // copy ctor
        GT_FUNCTION
        constexpr accessor_base(accessor_base const &other) : m_offsets(other.m_offsets) {}

        template < class... Ints,
            typename std::enable_if< sizeof...(Ints) <= Dim &&
                                         meta::conjunction< std::is_convertible< Ints, int_t >... >::value,
                int >::type = 0 >
        GT_FUNCTION constexpr explicit accessor_base(Ints... offsets)
            : m_offsets({offsets...}) {}

        template < ushort_t... Is,
            typename std::enable_if< sizeof...(Is) && meta::conjunction< is_dimension_index< Is >... >::value,
                int >::type = 0 >
        GT_FUNCTION explicit accessor_base(dimension< Is >... ds)
            : m_offsets({}) {
            noop{}(add_dimension(ds)...);
        }

        template < short_t Idx >
        GT_FUNCTION int_t constexpr get() const {
            GRIDTOOLS_STATIC_ASSERT(Idx < 0 || Idx <= Dim,
                "requested accessor index larger than the available "
                "dimensions. Maybe you made a mistake when setting the "
                "accessor dimensionality?");
            return m_offsets[Dim - 1 - Idx];
        }

        template < short_t Idx >
        GT_FUNCTION void set(uint_t offset_) {
            GRIDTOOLS_STATIC_ASSERT(Idx >= 0, "requested accessor index lower than zero");
            GRIDTOOLS_STATIC_ASSERT(
                Idx < 0 || Idx <= Dim, "requested accessor index larger than the available dimensions");
            m_offsets[Dim - 1 - Idx] = offset_;
        }
    };
}

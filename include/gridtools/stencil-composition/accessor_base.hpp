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

#include <tuple>
#include <type_traits>

#include "../common/array.hpp"
#include "../common/defs.hpp"
#include "../common/dimension.hpp"
#include "../common/generic_metafunctions/gt_integer_sequence.hpp"
#include "../common/generic_metafunctions/meta.hpp"
#include "../common/generic_metafunctions/type_traits.hpp"
#include "../common/host_device.hpp"
#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"

namespace gridtools {
    namespace _impl {
        template <ushort_t I>
        struct get_dimension_value_f {
            template <ushort_t J>
            GT_FUNCTION constexpr int_t operator()(dimension<J> src) const {
                return 0;
            }
            GT_FUNCTION constexpr int_t operator()(dimension<I> src) const { return src.value; }
        };

        template <ushort_t I>
        GT_FUNCTION constexpr int_t sum_dimensions() {
            return 0;
        }

        template <ushort_t I, class T, class... Ts>
        GT_FUNCTION constexpr int_t sum_dimensions(T src, Ts... srcs) {
            return get_dimension_value_f<I>{}(src) + sum_dimensions<I>(srcs...);
        }

        template <ushort_t Dim, ushort_t... Is, class... Ts>
        GT_FUNCTION constexpr array<int_t, Dim> make_offsets_impl(gt_integer_sequence<ushort_t, Is...>, Ts... srcs) {
            return {sum_dimensions<Is + 1>(srcs...)...};
        }

        template <ushort_t Dim, class... Ts>
        GT_FUNCTION constexpr array<int_t, Dim> make_offsets(Ts... srcs) {
            return make_offsets_impl<Dim>(make_gt_integer_sequence<ushort_t, Dim>{}, srcs...);
        }
    } // namespace _impl

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
    template <typename>
    struct accessor_base;

    // size_t... Enumeration instead of just the number of dimensions is to workaround bug #1040 (CUDA9.2, CUDA10)
    template <size_t... Enumeration>
    struct accessor_base<gt_index_sequence<Enumeration...>> {
        static constexpr size_t Dim = sizeof...(Enumeration);
        GRIDTOOLS_STATIC_ASSERT(Dim > 0, "dimension number must be positive");

        int m_offsets[Dim]; // c-array instead of std::array to workaround bug #1040

        static constexpr ushort_t n_dimensions = Dim;

        template <class... Ints,
            typename std::enable_if<sizeof...(Ints) <= Dim && conjunction<std::is_convertible<Ints, int_t>...>::value,
                int>::type = 0>
        GT_FUNCTION constexpr accessor_base(Ints... offsets) : m_offsets{offsets...} {}

        constexpr explicit accessor_base(accessor_base const &src) = default;

        template <ushort_t I, ushort_t... Is>
        GT_FUNCTION constexpr accessor_base(dimension<I> d, dimension<Is>... ds)
            : m_offsets{_impl::make_offsets<Dim>(d, ds...)[Enumeration]...} {}

        GT_FUNCTION constexpr int_t const &operator[](size_t i) const { return m_offsets[i]; }

        GT_FUNCTION
        int_t &operator[](size_t i) { return m_offsets[i]; }
    };

    template <int_t Idx, size_t... Enumeration>
    GT_FUNCTION constexpr int_t &get(accessor_base<gt_index_sequence<Enumeration...>> &acc) noexcept {
        GRIDTOOLS_STATIC_ASSERT(Idx >= 0, "requested accessor index lower than zero");
        GRIDTOOLS_STATIC_ASSERT(
            Idx < sizeof...(Enumeration), "requested accessor index larger than the available dimensions");
        return acc.m_offsets[Idx];
    }

    template <int_t Idx, size_t... Enumeration>
    GT_FUNCTION constexpr const int_t &get(const accessor_base<gt_index_sequence<Enumeration...>> &acc) noexcept {
        GRIDTOOLS_STATIC_ASSERT(Idx >= 0, "requested accessor index lower than zero");
        GRIDTOOLS_STATIC_ASSERT(
            Idx < sizeof...(Enumeration), "requested accessor index larger than the available dimensions");
        return acc.m_offsets[Idx];
    }

    template <int_t Idx, size_t... Enumeration>
    GT_FUNCTION constexpr int_t &&get(accessor_base<gt_index_sequence<Enumeration...>> &&acc) noexcept {
        return std::move(acc.m_offsets[Idx]);
    }

} // namespace gridtools

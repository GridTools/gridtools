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

#include "../common/array.hpp"
#include "../common/defs.hpp"
#include "../common/dimension.hpp"
#include "../common/host_device.hpp"
#include "../meta.hpp"

namespace gridtools {
    namespace accessor_base_impl_ {
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
        GT_FUNCTION constexpr array<int_t, Dim> make_offsets_impl(meta::integer_sequence<ushort_t, Is...>, Ts... srcs) {
            return {sum_dimensions<Is + 1>(srcs...)...};
        }

        template <ushort_t Dim, class... Ts>
        GT_FUNCTION constexpr array<int_t, Dim> make_offsets(Ts... srcs) {
            return make_offsets_impl<Dim>(meta::make_integer_sequence<ushort_t, Dim>{}, srcs...);
        }
    } // namespace accessor_base_impl_

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

    template <size_t Dim>
    struct accessor_base : array<int_t, Dim> {
#ifdef __INTEL_COMPILER
        int_t m_workaround = Dim;
#endif
        using base_t = array<int_t, Dim>;

        template <class... Ints,
            enable_if_t<sizeof...(Ints) <= Dim && conjunction<std::is_convertible<Ints, int_t>...>::value, int> = 0>
        GT_FUNCTION constexpr explicit accessor_base(Ints... offsets) : base_t{{offsets...}} {}

        GT_FUNCTION constexpr explicit accessor_base(base_t const &src) : base_t{src} {}

        template <ushort_t I, ushort_t... Is>
        GT_FUNCTION constexpr explicit accessor_base(dimension<I> d, dimension<Is>... ds)
            : base_t{accessor_base_impl_::make_offsets<Dim>(d, ds...)} {
            GT_STATIC_ASSERT((meta::is_set_fast<meta::list<dimension<I>, dimension<Is>...>>::value),
                "all dimensions should be of different indicies");
        }
    };

#ifdef __INTEL_COMPILER
    /* The Intel compiler does not want to vectorize when we use a real array here. */
    template <>
    struct accessor_base<3> {
        int_t data0, data1, data2;
        int_t m_workaround = 3;

        GT_FORCE_INLINE constexpr accessor_base(array<int_t, 3> const &a) : data0(a[0]), data1(a[1]), data2(a[2]) {}

        GT_FORCE_INLINE constexpr accessor_base(int_t data0 = {}, int_t data1 = {}, int_t data2 = {})
            : data0(data0), data1(data1), data2(data2) {}

        template <ushort_t I, ushort_t... Is>
        GT_FORCE_INLINE constexpr explicit accessor_base(dimension<I> d, dimension<Is>... ds)
            : accessor_base{accessor_base_impl_::make_offsets<3>(d, ds...)} {
            GT_STATIC_ASSERT((meta::is_set_fast<meta::list<dimension<I>, dimension<Is>...>>::value),
                "all dimensions should be of different indicies");
        }

        struct getter {
            template <size_t I>
            static GT_FORCE_INLINE constexpr enable_if_t<I == 0, int_t> get(accessor_base const &acc) noexcept {
                return acc.data0;
            }
            template <size_t I>
            static GT_FORCE_INLINE constexpr enable_if_t<I == 1, int_t> get(accessor_base const &acc) noexcept {
                return acc.data1;
            }
            template <size_t I>
            static GT_FORCE_INLINE constexpr enable_if_t<I == 2, int_t> get(accessor_base const &acc) noexcept {
                return acc.data2;
            }
        };
        friend getter tuple_getter(accessor_base const &) { return {}; }
        friend meta::list<int_t, int_t, int_t> tuple_to_types(accessor_base const &) { return {}; }
    };
#endif
} // namespace gridtools

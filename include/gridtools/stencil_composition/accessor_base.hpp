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
#include "../common/error.hpp"
#include "../common/functional.hpp"
#include "../common/host_device.hpp"
#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"

namespace gridtools {
    namespace accessor_base_impl_ {
        template <uint_t I>
        GT_FUNCTION int_t pick_dimension() {
            return 0;
        }

        template <uint_t I, class... Ts>
        GT_FUNCTION int_t pick_dimension(dimension<I> src, Ts &&...) {
            return src.value;
        }

        template <uint_t I, uint_t J, class... Ts, enable_if_t<I != J, int> = 0>
        GT_FUNCTION int_t pick_dimension(dimension<J> src, Ts... srcs) {
            return pick_dimension<I>(srcs...);
        }

        template <size_t>
        struct just_int {
            using type = int_t;
        };

        class check_all_zeros {
            bool m_dummy;

          public:
            template <class... Ts>
            GT_FUNCTION check_all_zeros(Ts... vals)
                : m_dummy{error_or_return(tuple_util::host_device::all_of(
                                              host_device::identity{}, array<bool, sizeof...(Ts)>{{(vals == 0)...}}),
                      false,
                      "unexpected non zero accessor offset")} {}
        };

        template <size_t Dim, uint_t I, enable_if_t<(I > Dim), int> = 0>
        GT_FUNCTION int_t out_of_range_dim(dimension<I> obj) {
            return obj.value;
        }

        template <size_t Dim, uint_t I, enable_if_t<(I <= Dim), int> = 0>
        GT_FUNCTION int_t out_of_range_dim(dimension<I>) {
            return 0;
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
     *
     *  TODO(anstaf) : check offsets against extent
     */

    template <size_t Dim, class = meta::make_index_sequence<Dim>>
    class accessor_base;

    template <size_t Dim, size_t... Is>
    class accessor_base<Dim, meta::index_sequence<Is...>> : public array<int_t, Dim> {
        using base_t = array<int_t, Dim>;

        template <class... Ts>
        GT_FUNCTION accessor_base(accessor_base_impl_::check_all_zeros, Ts... offsets)
            : base_t{{offsets...}} {}

      public:
        GT_FUNCTION accessor_base() : base_t{{}} {}

        template <class... Ts,
            enable_if_t<sizeof...(Ts) < Dim && conjunction<std::is_convertible<Ts, int_t>...>::value, int> = 0>
        GT_FUNCTION accessor_base(Ts... offsets) : base_t{{offsets...}} {}

        GT_FUNCTION accessor_base(base_t const &src) : base_t{src} {}

#ifndef NDEBUG
        template <class... Ts, enable_if_t<conjunction<std::is_convertible<Ts, int_t>...>::value, int> = 0>
        GT_FUNCTION accessor_base(typename accessor_base_impl_::just_int<Is>::type... offsets, Ts... zeros)
            : accessor_base{accessor_base_impl_::check_all_zeros{zeros...}, offsets...} {}

        template <uint_t J, uint_t... Js>
        GT_FUNCTION accessor_base(dimension<J> src, dimension<Js>... srcs)
            : accessor_base{accessor_base_impl_::check_all_zeros{accessor_base_impl_::out_of_range_dim<Dim>(src),
                                accessor_base_impl_::out_of_range_dim<Dim>(srcs)...},
                  accessor_base_impl_::pick_dimension<Is + 1>(src, srcs...)...} {
            GT_STATIC_ASSERT((meta::is_set_fast<meta::list<dimension<J>, dimension<Js>...>>::value),
                "all dimensions should be of different indicies");
        }
#else
        template <class... Ts, enable_if_t<conjunction<std::is_convertible<Ts, int_t>...>::value, int> = 0>
        GT_FUNCTION accessor_base(typename accessor_base_impl_::just_int<Is>::type... offsets, Ts...)
            : base_t{{offsets...}} {}

        template <uint_t J, uint_t... Js>
        GT_FUNCTION accessor_base(dimension<J> src, dimension<Js>... srcs)
            : base_t{{accessor_base_impl_::pick_dimension<Is + 1>(src, srcs...)...}} {
            GT_STATIC_ASSERT((meta::is_set_fast<meta::list<dimension<J>, dimension<Js>...>>::value),
                "all dimensions should be of different indicies");
        }
#endif
    };

    template <>
    class accessor_base<0, meta::index_sequence<>> : public tuple<> {
        template <class... Ts>
        GT_FUNCTION accessor_base(accessor_base_impl_::check_all_zeros) {}

      public:
        GT_DECLARE_DEFAULT_EMPTY_CTOR(accessor_base);

        GT_FUNCTION accessor_base(array<int_t, 0> const &) {}

#ifndef NDEBUG
        template <class... Ts, enable_if_t<conjunction<std::is_convertible<Ts, int_t>...>::value, int> = 0>
        GT_FUNCTION accessor_base(Ts... zeros)
            : accessor_base{accessor_base_impl_::check_all_zeros {
                  zeros...
              }} {}

        template <uint_t J, uint_t... Js>
        GT_FUNCTION accessor_base(dimension<J> zero, dimension<Js>... zeros)
            : accessor_base{accessor_base_impl_::check_all_zeros {
                  zero.value,
                  zeros.value...
              }} {}
#else
        template <class... Ts, enable_if_t<conjunction<std::is_convertible<Ts, int_t>...>::value, int> = 0>
        GT_FUNCTION accessor_base(Ts...) {}

        template <uint_t J, uint_t... Js>
        GT_FUNCTION accessor_base(dimension<J> zero, dimension<Js>... zeros) {}
#endif
    };
} // namespace gridtools

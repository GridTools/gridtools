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
#include <utility>

#include "../common/array.hpp"
#include "../common/defs.hpp"
#include "../common/dimension.hpp"
#include "../common/functional.hpp"
#include "../common/generic_metafunctions/accumulate.hpp"
#include "../common/host_device.hpp"
#include "../common/integral_constant.hpp"
#include "../common/tuple.hpp"
#include "../meta.hpp"
#include "accessor_intent.hpp"
#include "extent.hpp"
#include "is_accessor.hpp"
#include "location_type.hpp"

namespace gridtools {
    namespace accessor_base_impl_ {
        template <uint_t I>
        GT_FUNCTION GT_CONSTEXPR int_t pick_dimension() {
            return 0;
        }

        template <uint_t I, class... Ts>
        GT_FUNCTION GT_CONSTEXPR int_t pick_dimension(dimension<I> src, Ts &&...) {
            return src.value;
        }

        template <uint_t I, uint_t J, class... Ts, std::enable_if_t<I != J, int> = 0>
        GT_FUNCTION GT_CONSTEXPR int_t pick_dimension(dimension<J> src, Ts... srcs) {
            return pick_dimension<I>(srcs...);
        }

        template <size_t>
        struct just_int {
            using type = int_t;
        };

        template <class... Ts>
        using are_ints = conjunction<std::is_convertible<Ts, int_t>...>;

        template <size_t Dim, uint_t I, std::enable_if_t<(I > Dim), int> = 0>
        GT_FUNCTION GT_CONSTEXPR int_t out_of_range_dim(dimension<I> obj) {
            return obj.value;
        }

        template <size_t Dim, uint_t I, std::enable_if_t<(I <= Dim), int> = 0>
        GT_FUNCTION GT_CONSTEXPR int_t out_of_range_dim(dimension<I>) {
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

    template <uint_t Id, intent Intent, class Extent, size_t Dim, class = std::make_index_sequence<Dim>>
    class accessor_base;

    template <uint_t Id, intent Intent, class Extent, size_t Dim, size_t... Is>
    class accessor_base<Id, Intent, Extent, Dim, std::index_sequence<Is...>> : public array<int_t, Dim> {
        using base_t = array<int_t, Dim>;

      public:
        using index_t = integral_constant<uint_t, Id>;
        static constexpr intent intent_v = Intent;
        using extent_t = Extent;
        using location_t = enumtype::default_location_type;

        GT_FUNCTION GT_CONSTEXPR accessor_base() : base_t({}) {}

        GT_FUNCTION GT_CONSTEXPR accessor_base(base_t src) : base_t(wstd::move(src)) {}

        template <class... Ts,
            std::enable_if_t<sizeof...(Ts) < Dim && conjunction<std::is_convertible<Ts, int_t>...>::value, int> = 0>
        GT_FUNCTION GT_CONSTEXPR accessor_base(Ts... offsets) : base_t({offsets...}) {}

        template <class... Ts, std::enable_if_t<accessor_base_impl_::are_ints<Ts...>::value, int> = 0>
        GT_FUNCTION GT_CONSTEXPR accessor_base(typename accessor_base_impl_::just_int<Is>::type... offsets, Ts... zeros)
            : base_t({offsets...}) {
            assert(accumulate(logical_and(), true, (zeros == 0)...));
        }

        template <uint_t J, uint_t... Js>
        GT_FUNCTION GT_CONSTEXPR accessor_base(dimension<J> src, dimension<Js>... srcs)
            : base_t({accessor_base_impl_::pick_dimension<Is + 1>(src, srcs...)...}) {
            static_assert(meta::is_set_fast<meta::list<dimension<J>, dimension<Js>...>>::value,
                "all dimensions should be of different indicies");
            assert(accessor_base_impl_::out_of_range_dim<Dim>(src) == 0);
            assert(accumulate(logical_and(), true, (accessor_base_impl_::out_of_range_dim<Dim>(srcs) == 0)...));
        }
    };

    template <uint_t Id, class Extent, intent Intent>
    class accessor_base<Id, Intent, Extent, 0, std::index_sequence<>> : public tuple<> {
      public:
        static_assert(std::is_same<Extent, extent<>>::value, GT_INTERNAL_ERROR);

        using index_t = integral_constant<uint_t, Id>;
        static constexpr intent intent_v = Intent;
        using extent_t = Extent;
        using location_t = enumtype::default_location_type;

        GT_DECLARE_DEFAULT_EMPTY_CTOR(accessor_base);

        template <class... Ts, std::enable_if_t<accessor_base_impl_::are_ints<Ts...>::value, int> = 0>
        GT_FUNCTION GT_CONSTEXPR accessor_base(Ts... zeros) {
            assert(accumulate(logical_and(), true, (zeros == 0)...));
        }

        template <uint_t J, uint_t... Js>
        GT_FUNCTION GT_CONSTEXPR accessor_base(dimension<J> zero, dimension<Js>... zeros) {
            assert(zero.value == 0);
            assert(accumulate(logical_and(), true, (zeros.value == 0)...));
        }
    };

    template <uint_t ID, intent Intent, typename Extent, size_t Number>
    meta::repeat_c<Number, int_t> tuple_to_types(accessor_base<ID, Intent, Extent, Number> const &);

    template <uint_t ID, intent Intent, typename Extent, size_t Number>
    meta::always<accessor_base<ID, Intent, Extent, Number>> tuple_from_types(
        accessor_base<ID, Intent, Extent, Number> const &);

    template <uint_t ID, intent Intent, typename Extent, size_t Number, class Seq>
    struct is_accessor<accessor_base<ID, Intent, Extent, Number, Seq>> : std::true_type {};
} // namespace gridtools

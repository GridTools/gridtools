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
#include "../common/tuple_util.hpp"
#include "../meta.hpp"

namespace gridtools {
#ifdef __INTEL_COMPILER
    namespace _impl {
        /* Pseudo-array class, only used for the Intel compiler which has problems vectorizing the accessor_base
         * class with a normal array member. Currently only the 3D case is specialized to allow for good vectorization
         * in the most common case. */
        template <std::size_t Dim>
        struct pseudo_array_type {
            using type = array<int_t, Dim>;
        };

        struct pseudo_array_type<3> {
            struct type {
                T data0, data1, data2;

                constexpr type(array<int_t, 3> const &a) : data0(get<0>(a)), data1(get<1>(a)), data2(get<2>(a)) {}

                constexpr type(int_t data0 = {}, int_t data1 = {}, int_t data2 = {})
                    : data0(data0), data1(data1), data2(data2) {}

                GT_FUNCTION T &operator[](std::size_t i) { return (&data0)[i]; }

                struct getter {
                    template <size_t I>
                    static GT_FUNCTION constexpr enable_if_t<I == 0, int_t> get(type const &acc) noexcept {
                        return arr.data0;
                    }
                    template <size_t I>
                    static GT_FUNCTION constexpr enable_if_t<I == 1, int_t> get(type const &acc) noexcept {
                        return arr.data1;
                    }
                    template <size_t I>
                    static GT_FUNCTION constexpr enable_if_t<I == 2, int_t> get(type const &acc) noexcept {
                        return arr.data2;
                    }
                };
                friend getter tuple_getter(type const &) { return {}; }
                friend meta::list<int_t, int_t, int_t> tuple_to_types(type const &) { return {}; }
            };
        };
    } // namespace _impl
#endif

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
        GT_FUNCTION constexpr array<int_t, Dim> make_offsets_impl(meta::integer_sequence<ushort_t, Is...>, Ts... srcs) {
            return {sum_dimensions<Is + 1>(srcs...)...};
        }

        template <ushort_t Dim, class... Ts>
        GT_FUNCTION constexpr array<int_t, Dim> make_offsets(Ts... srcs) {
            return make_offsets_impl<Dim>(meta::make_integer_sequence<ushort_t, Dim>{}, srcs...);
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
    template <ushort_t Dim>
    class accessor_base {
        GT_STATIC_ASSERT(Dim > 0, "dimension number must be positive");

#ifdef __INTEL_COMPILER
        /* The Intel compiler does not want to vectorize when we use a real array here. */
        using offsets_t = typename _impl::pseudo_array_type<Dim>::type;
        offsets_t m_offsets;
        /* The Intel compiler likes to generate calls to memset if we don't have this additional member.*/
        int_t m_workaround;
#else
        using offsets_t = array<int_t, Dim>;
        offsets_t m_offsets;
#endif
        struct getter {
            template <size_t I>
            static GT_FUNCTION constexpr int_t get(accessor_base const &acc) noexcept {
                GT_STATIC_ASSERT(I >= 0, "requested accessor index lower than zero");
                GT_STATIC_ASSERT(I < Dim, "requested accessor index larger than the available dimensions");
                return tuple_util::host_device::get<I>(acc.m_offsets);
            }
        };
        friend getter tuple_getter(accessor_base const &) { return {}; }
        friend GT_META_CALL(meta::repeat_c, (Dim, int_t)) tuple_to_types(accessor_base const &) { return {}; }

      public:
        static constexpr ushort_t n_dimensions = Dim;

        template <class... Ints,
            enable_if_t<sizeof...(Ints) <= Dim && conjunction<std::is_convertible<Ints, int_t>...>::value, int> = 0>
        GT_FUNCTION constexpr explicit accessor_base(Ints... offsets) : m_offsets {
            offsets...
        }
#ifdef __INTEL_COMPILER
        , m_workaround(Dim)
#endif
        {
        }

        GT_FUNCTION constexpr explicit accessor_base(offsets_t const &src)
            : m_offsets(src)
#ifdef __INTEL_COMPILER
              ,
              m_workaround(Dim)
#endif
        {
        }

        template <ushort_t I, ushort_t... Is>
        GT_FUNCTION constexpr explicit accessor_base(dimension<I> d, dimension<Is>... ds)
            : m_offsets(_impl::make_offsets<Dim>(d, ds...))
#ifdef __INTEL_COMPILER
              ,
              m_workaround(Dim)
#endif
        {
            GT_STATIC_ASSERT((meta::is_set<meta::list<dimension<I>, dimension<Is>...>>::value),
                "all dimensions should be of different indicies");
        }
    };
} // namespace gridtools

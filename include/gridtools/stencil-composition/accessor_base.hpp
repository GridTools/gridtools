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
#ifdef __INTEL_COMPILER
    namespace _impl {
        /* Pseudo-array class, only used for the Intel compiler which has problems vectorizing the accessor_base
         * class with a normal array member. Currently only the 3D case is specialized to allow for good vectorization
         * in the most common case. */
        template <typename T, std::size_t Dim>
        struct pseudo_array_type {
            using type = array<T, Dim>;
        };

        template <typename T>
        struct pseudo_array_type<T, 3> {
            struct type {
                T data0, data1, data2;

                constexpr type(array<T, 3> const &a) : data0(get<0>(a)), data1(get<1>(a)), data2(get<2>(a)) {}

                constexpr type() : data0(0), data1(0), data2(0) {}

                constexpr type(T const &data0, T const &data1, T const &data2)
                    : data0(data0), data1(data1), data2(data2) {}

                GT_FUNCTION T &operator[](std::size_t i) { return (&data0)[i]; }
            };
        };
    } // namespace _impl

    template <std::size_t Idx>
    GT_FUNCTION constexpr typename std::enable_if<Idx == 0, int_t const &>::type get(
        typename _impl::pseudo_array_type<int_t, 3>::type const &arr) noexcept {
        return arr.data0;
    }
    template <std::size_t Idx>
    GT_FUNCTION constexpr typename std::enable_if<Idx == 1, int_t const &>::type get(
        typename _impl::pseudo_array_type<int_t, 3>::type const &arr) noexcept {
        return arr.data1;
    }
    template <std::size_t Idx>
    GT_FUNCTION constexpr typename std::enable_if<Idx == 2, int_t const &>::type get(
        typename _impl::pseudo_array_type<int_t, 3>::type const &arr) noexcept {
        return arr.data2;
    }
    template <std::size_t Idx>
    GT_FUNCTION constexpr typename std::enable_if<Idx == 0, int_t &>::type get(
        typename _impl::pseudo_array_type<int_t, 3>::type &arr) noexcept {
        return arr.data0;
    }
    template <std::size_t Idx>
    GT_FUNCTION constexpr typename std::enable_if<Idx == 1, int_t &>::type get(
        typename _impl::pseudo_array_type<int_t, 3>::type &arr) noexcept {
        return arr.data1;
    }
    template <std::size_t Idx>
    GT_FUNCTION constexpr typename std::enable_if<Idx == 2, int_t &>::type get(
        typename _impl::pseudo_array_type<int_t, 3>::type &arr) noexcept {
        return arr.data2;
    }

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
    //    template <ushort_t Dim>
    //    using offsets_t = int[Dim];

    //        template <ushort_t RequestedI, >

    //    template<ushort_t RequestedI

    //    template <ushort_t...>
    //    struct my_impl;
    //
    //    template <ushort_t RequestedI, ushort_t I, ushort... Is>
    //    struct my_impl<RequestedI, I, Is...> {
    //        GT_FUNCTION constexpr enable_if_t<RequestedI == I, int> operator()() const { return 0; }
    //        //        using type = typename std::conditional<RequestedI == I, int, typename my_impl<RequestedI,
    //        //        Is...>::type>::type;
    //    };
    //
    ////    template <ushort_t RequestedI, ushort... Is>
    ////    struct my_impl<RequestedI, RequestedI, Is...> {
    ////        GT_FUNCTION constexpr int operator()() const { return 0; }
    ////    };
    //
    //    template <ushort_t RequestedI>
    //    struct my_impl<RequestedI> {
    //        GT_FUNCTION constexpr int operator()() const { return 0; }
    //    };

    //    template <ushort_t RequestedI>
    //    struct init_from_dimensions {
    //        template <ushort_t I, ushort_t... Is>
    //        GT_FUNCTION constexpr enable_if_t<I == RequestedI, int> operator()(
    //            dimension<I> const &d, dimension<Is> const &... ds) const {
    //            return d.value;
    //        }
    //        template <ushort_t I, ushort_t... Is>
    //        GT_FUNCTION constexpr enable_if_t<I != RequestedI, int> operator()(
    //            dimension<I> const &d, dimension<Is> const &... ds) const {
    //            return operator()(ds...);
    //        }
    //
    //        GT_FUNCTION constexpr int operator()() const { return 0; }
    //    };
    //    template <ushort_t RequestedI, typename... Dims>
    //    GT_FUNCTION constexpr int my_func(Dims const &... ds) {
    //        return RequestedI;
    //    }

    template <typename>
    struct accessor_base;

    template <size_t... Enumeration>
    struct accessor_base<gt_index_sequence<Enumeration...>> {
        static constexpr size_t Dim = sizeof...(Enumeration);
        GRIDTOOLS_STATIC_ASSERT(Dim > 0, "dimension number must be positive");

#ifdef __INTEL_COMPILER
        /* The Intel compiler does not want to vectorize when we use a real array here. */
        using offsets_t = typename _impl::pseudo_array_type<int_t, Dim>::type;
        offsets_t m_offsets;
        /* The Intel compiler likes to generate calls to memset if we don't have this additional member.*/
        int_t m_workaround;
#else
        //        typedef int offsets_t[Dim];
        //        offsets_t<Dim> m_offsets;
        int m_offsets[Dim];
#endif

        static constexpr ushort_t n_dimensions = Dim;

        GT_FUNCTION constexpr int *const &offsets() const { return m_offsets; }
        GT_FUNCTION int *&offsets() { return m_offsets; }

        template <class... Ints,
            typename std::enable_if<sizeof...(Ints) <= Dim && conjunction<std::is_convertible<Ints, int_t>...>::value,
                int>::type = 0>
        GT_FUNCTION constexpr accessor_base(Ints... offsets) : m_offsets{offsets...} {}

        constexpr accessor_base(accessor_base const &src) = default;
        //        GT_FUNCTION constexpr explicit accessor_base(offsets_t const &src)
        //            : m_offsets(src)
        //              //#ifdef __INTEL_COMPILER
        //              ,
        //              m_workaround(Dim)
        //        //#endif
        //        {}

        struct fill {
            template <ushort_t I>
            GT_FUNCTION void operator()(dimension<I> d) {
                m_offsets[I] = d.value;
            }
        };

        template <ushort_t I, ushort_t... Is>
        GT_FUNCTION constexpr accessor_base(dimension<I> d, dimension<Is>... ds)
            : m_offsets{_impl::make_offsets<Dim>(d, ds...)[Enumeration]...} {}
        //                    : m_offsets{/*init_from_dimensions<Is>{}(ds...)... my_func<Is>(ds...)...*/} {}

        //        template <ushort_t I, ushort_t... Is>
        //        GT_FUNCTION accessor_base(dimension<I> d, dimension<Is>... ds) {
        //            tuple_util::host_device::for_each(fill{}, std::tuple<dimension<I>,
        //            dimension<Is>...>{d, ds...});
        //        }
        //            : m_offsets{/*init_from_dimensions<Is>{}(ds...)... my_func<Is>(ds...)...*/} {}

        //        template <ushort_t I, ushort_t... Is>
        //        GT_FUNCTION constexpr explicit accessor_base(dimension<I> d, dimension<Is>... ds)
        //            : m_offsets(_impl::make_offsets<Dim>(d, ds...))
        //#ifdef __INTEL_COMPILER
        //              ,
        //              m_workaround(Dim)
        //#endif
        //        {
        //            GRIDTOOLS_STATIC_ASSERT((meta::is_set<meta::list<dimension<I>,
        //            dimension<Is>...>>::value),
        //                "all dimensions should be of different indicies");
        //        }

        GT_FUNCTION constexpr int_t const &operator[](size_t i) const { return m_offsets[i]; }

        GT_FUNCTION
        int_t &operator[](size_t i) { return m_offsets[i]; }
    };

    template <int_t Idx, size_t... Enumeration>
    GT_FUNCTION constexpr int_t &get(accessor_base<gt_index_sequence<Enumeration...>> &acc) noexcept {
        GRIDTOOLS_STATIC_ASSERT(Idx >= 0, "requested accessor index lower than zero");
        GRIDTOOLS_STATIC_ASSERT(
            Idx < sizeof...(Enumeration), "requested accessor index larger than the available dimensions");
        return acc.m_offsets[Idx]; // get<Idx>(acc.offsets());
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

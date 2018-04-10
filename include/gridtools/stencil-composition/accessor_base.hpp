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

#include <type_traits>

#include "../common/array.hpp"
#include "../common/defs.hpp"
#include "../common/dimension.hpp"
#include "../common/host_device.hpp"
#include "../common/generic_metafunctions/gt_integer_sequence.hpp"
#include "../common/generic_metafunctions/meta.hpp"

namespace gridtools {

    namespace _impl {

#ifdef __INTEL_COMPILER
        /* Pseudo-array class, only used for the Intel compiler which has problems vectorizing the accessor_base
         * class with a normal array member and even produces incorrect code. */
        template < typename T, std::size_t Dim >
        struct pseudo_array {
            GRIDTOOLS_STATIC_ASSERT(!Dim, "Pseudo array not implemented for the given number of dimensions");
        };

        template < typename T >
        struct pseudo_array< T, 2 > {
            T data0, data1;

            GT_FUNCTION constexpr pseudo_array(array< T, 2 > const &a)
                : data0(a.template get< 0 >()), data1(a.template get< 1 >()) {}

            GT_FUNCTION constexpr pseudo_array(T data0 = 0, T data1 = 0) : data0(data0), data1(data1) {}

            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 0, T const & >::type get() const {
                return data0;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 1, T const & >::type get() const {
                return data1;
            }

            GT_FUNCTION T &operator[](std::size_t i) { return (&data0)[i]; }
        };

        template < typename T >
        struct pseudo_array< T, 3 > {
            T data0, data1, data2;

            GT_FUNCTION constexpr pseudo_array(array< T, 3 > const &a)
                : data0(a.template get< 0 >()), data1(a.template get< 1 >()), data2(a.template get< 2 >()) {}

            GT_FUNCTION constexpr pseudo_array(T data0 = 0, T data1 = 0, T data2 = 0)
                : data0(data0), data1(data1), data2(data2) {}

            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 0, T const & >::type get() const {
                return data0;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 1, T const & >::type get() const {
                return data1;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 2, T const & >::type get() const {
                return data2;
            }

            GT_FUNCTION T &operator[](std::size_t i) { return (&data0)[i]; }
        };

        template < typename T >
        struct pseudo_array< T, 4 > {
            T data0, data1, data2, data3;

            GT_FUNCTION constexpr pseudo_array(array< T, 4 > const &a)
                : data0(a.template get< 0 >()), data1(a.template get< 1 >()), data2(a.template get< 2 >()),
                  data3(a.template get< 3 >()) {}

            GT_FUNCTION constexpr pseudo_array(T data0, T data1, T data2, T data3)
                : data0(data0), data1(data1), data2(data2), data3(data3) {}

            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 0, T const & >::type get() const {
                return data0;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 1, T const & >::type get() const {
                return data1;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 2, T const & >::type get() const {
                return data2;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 3, T const & >::type get() const {
                return data3;
            }

            GT_FUNCTION T &operator[](std::size_t i) { return (&data0)[i]; }
        };

        template < typename T >
        struct pseudo_array< T, 5 > {
            T data0, data1, data2, data3, data4;

            GT_FUNCTION constexpr pseudo_array(array< T, 5 > const &a)
                : data0(a.template get< 0 >()), data1(a.template get< 1 >()), data2(a.template get< 2 >()),
                  data3(a.template get< 3 >()), data4(a.template get< 4 >()) {}

            GT_FUNCTION constexpr pseudo_array(T data0 = 0, T data1 = 0, T data2 = 0, T data3 = 0, T data4 = 0)
                : data0(data0), data1(data1), data2(data2), data3(data3), data4(data4) {}

            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 0, T const & >::type get() const {
                return data0;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 1, T const & >::type get() const {
                return data1;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 2, T const & >::type get() const {
                return data2;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 3, T const & >::type get() const {
                return data3;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 4, T const & >::type get() const {
                return data4;
            }

            GT_FUNCTION T &operator[](std::size_t i) { return (&data0)[i]; }
        };

        template < typename T >
        struct pseudo_array< T, 6 > {
            T data0, data1, data2, data3, data4, data5;

            GT_FUNCTION constexpr pseudo_array(array< T, 6 > const &a)
                : data0(a.template get< 0 >()), data1(a.template get< 1 >()), data2(a.template get< 2 >()),
                  data3(a.template get< 3 >()), data4(a.template get< 4 >()), data5(a.template get< 5 >()) {}

            GT_FUNCTION constexpr pseudo_array(
                T data0 = 0, T data1 = 0, T data2 = 0, T data3 = 0, T data4 = 0, T data5 = 0)
                : data0(data0), data1(data1), data2(data2), data3(data3), data4(data4), data5(data5) {}

            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 0, T const & >::type get() const {
                return data0;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 1, T const & >::type get() const {
                return data1;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 2, T const & >::type get() const {
                return data2;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 3, T const & >::type get() const {
                return data3;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 4, T const & >::type get() const {
                return data4;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 5, T const & >::type get() const {
                return data5;
            }

            GT_FUNCTION T &operator[](std::size_t i) { return (&data0)[i]; }
        };

        template < typename T >
        struct pseudo_array< T, 7 > {
            T data0, data1, data2, data3, data4, data5, data6;

            GT_FUNCTION constexpr pseudo_array(array< T, 7 > const &a)
                : data0(a.template get< 0 >()), data1(a.template get< 1 >()), data2(a.template get< 2 >()),
                  data3(a.template get< 3 >()), data4(a.template get< 4 >()), data5(a.template get< 5 >()),
                  data6(a.template get< 6 >()) {}

            GT_FUNCTION constexpr pseudo_array(
                T data0 = 0, T data1 = 0, T data2 = 0, T data3 = 0, T data4 = 0, T data5 = 0, T data6 = 0)
                : data0(data0), data1(data1), data2(data2), data3(data3), data4(data4), data5(data5), data6(data6) {}

            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 0, T const & >::type get() const {
                return data0;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 1, T const & >::type get() const {
                return data1;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 2, T const & >::type get() const {
                return data2;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 3, T const & >::type get() const {
                return data3;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 4, T const & >::type get() const {
                return data4;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 5, T const & >::type get() const {
                return data5;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 6, T const & >::type get() const {
                return data6;
            }

            GT_FUNCTION T &operator[](std::size_t i) { return (&data0)[i]; }
        };

        template < typename T >
        struct pseudo_array< T, 15 > {
            T data0, data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13,
                data14;

            GT_FUNCTION constexpr pseudo_array(array< T, 15 > const &a)
                : data0(a.template get< 0 >()), data1(a.template get< 1 >()), data2(a.template get< 2 >()),
                  data3(a.template get< 3 >()), data4(a.template get< 4 >()), data5(a.template get< 5 >()),
                  data6(a.template get< 6 >()), data7(a.template get< 7 >()), data8(a.template get< 8 >()),
                  data9(a.template get< 9 >()), data10(a.template get< 10 >()), data11(a.template get< 11 >()),
                  data12(a.template get< 12 >()), data13(a.template get< 13 >()), data14(a.template get< 14 >()) {}

            GT_FUNCTION constexpr pseudo_array(T data0 = 0,
                T data1 = 0,
                T data2 = 0,
                T data3 = 0,
                T data4 = 0,
                T data5 = 0,
                T data6 = 0,
                T data7 = 0,
                T data8 = 0,
                T data9 = 0,
                T data10 = 0,
                T data11 = 0,
                T data12 = 0,
                T data13 = 0,
                T data14 = 0)
                : data0(data0), data1(data1), data2(data2), data3(data3), data4(data4), data5(data5), data6(data6),
                  data7(data7), data8(data8), data9(data9), data10(data10), data11(data11), data12(data12),
                  data13(data13), data14(data14) {}

            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 0, T const & >::type get() const {
                return data0;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 1, T const & >::type get() const {
                return data1;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 2, T const & >::type get() const {
                return data2;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 3, T const & >::type get() const {
                return data3;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 4, T const & >::type get() const {
                return data4;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 5, T const & >::type get() const {
                return data5;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 6, T const & >::type get() const {
                return data6;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 7, T const & >::type get() const {
                return data7;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 8, T const & >::type get() const {
                return data8;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 9, T const & >::type get() const {
                return data9;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 10, T const & >::type get() const {
                return data10;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 11, T const & >::type get() const {
                return data11;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 12, T const & >::type get() const {
                return data12;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 13, T const & >::type get() const {
                return data13;
            }
            template < std::size_t Idx >
            GT_FUNCTION constexpr typename std::enable_if< Idx == 14, T const & >::type get() const {
                return data14;
            }

            GT_FUNCTION T &operator[](std::size_t i) { return (&data0)[i]; }
        };
#endif

        template < ushort_t I >
        struct get_dimension_value_f {
            template < ushort_t J >
            GT_FUNCTION constexpr int_t operator()(dimension< J > src) const {
                return 0;
            }
            GT_FUNCTION constexpr int_t operator()(dimension< I > src) const { return src.value; }
        };

        template < ushort_t I >
        GT_FUNCTION constexpr int_t sum_dimensions() {
            return 0;
        }

        template < ushort_t I, class T, class... Ts >
        GT_FUNCTION constexpr int_t sum_dimensions(T src, Ts... srcs) {
            return get_dimension_value_f< I >{}(src) + sum_dimensions< I >(srcs...);
        }

        template < ushort_t Dim, ushort_t... Is, class... Ts >
        GT_FUNCTION constexpr array< int_t, Dim > make_offsets_impl(
            gt_integer_sequence< ushort_t, Is... >, Ts... srcs) {
            return {sum_dimensions< Is + 1 >(srcs...)...};
        }

        template < ushort_t Dim, class... Ts >
        GT_FUNCTION constexpr array< int_t, Dim > make_offsets(Ts... srcs) {
            return make_offsets_impl< Dim >(make_gt_integer_sequence< ushort_t, Dim >{}, srcs...);
        }
    }

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
        GRIDTOOLS_STATIC_ASSERT(Dim > 0, "dimension number must be positive");

#ifdef __INTEL_COMPILER
        /* The Intel compiler does not want to vectorize when we use a real array here. */
        using offsets_t = _impl::pseudo_array< int_t, Dim >;
        offsets_t m_offsets;
        /* The Intel compiler likes to generate calls to memset if we don't have this additional member.*/
        int_t m_workaround;
#else
        using offsets_t = array< int_t, Dim >;
        offsets_t m_offsets;
#endif

      public:
        static const ushort_t n_dimensions = Dim;

        template < class... Ints,
            typename std::enable_if< sizeof...(Ints) <= Dim &&
                                         meta::conjunction< std::is_convertible< Ints, int_t >... >::value,
                int >::type = 0 >
        GT_FUNCTION constexpr explicit accessor_base(Ints... offsets)
            : m_offsets({offsets...})
#ifdef __INTEL_COMPILER
              ,
              m_workaround(Dim)
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

        template < ushort_t I, ushort_t... Is >
        GT_FUNCTION constexpr explicit accessor_base(dimension< I > d, dimension< Is >... ds)
            : m_offsets(_impl::make_offsets< Dim >(d, ds...))
#ifdef __INTEL_COMPILER
              ,
              m_workaround(Dim)
#endif
        {
            GRIDTOOLS_STATIC_ASSERT((meta::is_set< meta::list< dimension< I >, dimension< Is >... > >::value),
                "all dimensions should be of different indicies");
        }

        template < short_t Idx >
        GT_FUNCTION int_t constexpr get() const {
            GRIDTOOLS_STATIC_ASSERT(Idx >= 0, "requested accessor index lower than zero");
            GRIDTOOLS_STATIC_ASSERT(Idx < Dim, "requested accessor index larger than the available dimensions");
            return m_offsets.template get< Dim - 1 - Idx >();
        }

        template < short_t Idx >
        GT_FUNCTION void set(uint_t offset_) {
            GRIDTOOLS_STATIC_ASSERT(Idx >= 0, "requested accessor index lower than zero");
            GRIDTOOLS_STATIC_ASSERT(Idx < Dim, "requested accessor index larger than the available dimensions");
            m_offsets[Dim - 1 - Idx] = offset_;
        }
    };
}

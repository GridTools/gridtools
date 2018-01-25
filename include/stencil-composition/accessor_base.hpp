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
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/type_traits/integral_constant.hpp>

#include "../gridtools.hpp"

#include "../storage/storage-facility.hpp"

#include "../common/offset_tuple_mixed.hpp"
#include "extent.hpp"
#include "arg_fwd.hpp"
#include "accessor_metafunctions.hpp"

namespace gridtools {

    // forward declaration
    template < int_t Index, int_t NDim >
    struct offset_tuple;

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
     * @tparam Extend Bounds over which the function access the argument
     */
    template < uint_t I, enumtype::intend Intend, typename Extend, ushort_t Dim >
    struct accessor_base {

        // typedef useful when unnecessary indirections are used
        typedef accessor_base< I, Intend, Extend, Dim > type;
        template < uint_t II, enumtype::intend It, typename R, ushort_t D >
        friend std::ostream &operator<<(std::ostream &s, accessor_base< II, It, R, D > const &x);

        typedef accessor_base< I, Intend, Extend, Dim > base_t;
        static const ushort_t n_dimensions = Dim;

        typedef static_uint< I > index_t;
        typedef enumtype::enum_type< enumtype::intend, Intend > intend_t;
        typedef Extend extent_t;
        typedef offset_tuple< n_dimensions, n_dimensions > offset_tuple_t;

      private:
        offset_tuple_t m_offsets;

        // friend of itself with different indices
        template < uint_t, enumtype::intend, typename, ushort_t >
        friend struct accessor_base;

      public:
#ifdef GT_NO_CONSTEXPR_ACCESSES
#define constexpr
#endif

        /**@brief Default constructor
           NOTE: the following constructor when used with the brace initializer produces with nvcc a considerable amount
           of extra instructions (gcc 4.8.2), and degrades the performances (which is probably a compiler bug, I
           couldn't reproduce it on a small test).*/
        GT_FUNCTION
        constexpr explicit accessor_base() : m_offsets() {}

        template < size_t ArrayDim,
            typename... Dimensions,
            typename Dummy = typename all_dimensions< dimension< 0 >, Dimensions... >::type >
        GT_FUNCTION constexpr explicit accessor_base(array< int_t, ArrayDim > const &offsets, Dimensions... d)
            : m_offsets(0, offsets, d...) {}

        // move ctor
        GT_FUNCTION
        constexpr accessor_base(const type &&other) : m_offsets(std::move(other.m_offsets)) {}

        // move ctor from another accessor_base with different index
        template < uint_t OtherIndex >
        GT_FUNCTION constexpr accessor_base(accessor_base< OtherIndex, Intend, Extend, Dim > &&other)
            : m_offsets(std::move(other.m_offsets)) {}

        // copy ctor
        GT_FUNCTION
        constexpr accessor_base(type const &other) : m_offsets(other.m_offsets) {}

        // copy ctor from another accessor_base with different index
        template < uint_t OtherIndex >
        GT_FUNCTION constexpr accessor_base(const accessor_base< OtherIndex, Intend, Extend, Dim > &other)
            : m_offsets(other.m_offsets) {}

        /**@brief constructor taking the dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'dimension' is a
           language keyword used at the interface level.
        */
        template < typename... Indices, typename Dummy = all_accessor_ctr_args< Indices... > >
        GT_FUNCTION constexpr accessor_base(Indices... x)
            : m_offsets(x...) {
            GRIDTOOLS_STATIC_ASSERT(sizeof...(x) <= n_dimensions,
                "the number of arguments passed to the offset_tuple constructor exceeds the number of space dimensions "
                "of the storage. Check that you are not accessing a non existing dimension, or increase the dimension "
                "D of the accessor (accessor<Id, extent, D>)");
        }

        template < typename First,
            typename... Rest,
            typename T = typename boost::enable_if_c< accumulate(
                logical_and(), is_dimension< First >::type::value, is_dimension< Rest >::type::value...) >::type >

        GT_FUNCTION constexpr accessor_base(First f, Rest... x)
            : m_offsets(f, x...) {
            GRIDTOOLS_STATIC_ASSERT(
                accumulate(logical_and(), (First::index <= n_dimensions), (Rest::index <= n_dimensions)...),
                "trying to access a too high dimension for accessor");
            GRIDTOOLS_STATIC_ASSERT(sizeof...(x) <= n_dimensions - 1,
                "the number of arguments passed to the offset_tuple constructor exceeds the number of space dimensions "
                "of the storage. Check that you are not accessing a non existing dimension, or increase the dimension "
                "D of the accessor (accessor<Id, extent, D>)");
        }

#ifdef GT_NO_CONSTEXPR_ACCESSES
#undef constexpr
#endif

        static void info() { std::cout << "Arg_type storage with index " << I << " and extent " << Extend() << " "; }

        template < short_t Idx >
        GT_FUNCTION constexpr bool end() const {
            return true;
        }

        template < short_t Idx >
        GT_FUNCTION int_t constexpr get() const {
            GRIDTOOLS_STATIC_ASSERT(Idx < 0 || Idx <= n_dimensions,
                "requested accessor index larger than the available "
                "dimensions. Maybe you made a mistake when setting the "
                "accessor dimensionality?");
            return m_offsets.template get< Idx >();
        }

        template < short_t Idx >
        GT_FUNCTION void set(uint_t offset_) {
            GRIDTOOLS_STATIC_ASSERT(Idx >= 0, "requested accessor index lower than zero");
            GRIDTOOLS_STATIC_ASSERT(
                Idx < 0 || Idx <= n_dimensions, "requested accessor index larger than the available dimensions");
            m_offsets.template set< Idx >(offset_);
        }

        GT_FUNCTION
        offset_tuple_t &offsets() { return m_offsets; }

        GT_FUNCTION
        constexpr const offset_tuple_t &offsets() const { return m_offsets; }

        template < ushort_t Idx >
        GT_FUNCTION void increment(int_t offset_) {
            m_offsets.template increment< Idx >(offset_);
        }
    };

    /**
     * Printing type information for debug purposes
     * @param s The ostream
     * @param n/a Type selector for offset_tuple
     * @return ostream
     */
    template < uint_t I, enumtype::intend It, typename R, ushort_t D >
    std::ostream &operator<<(std::ostream &s, accessor_base< I, It, R, D > const &x) {
        s << "[ offset_tuple< " << I << ", " << R() << ", " << It << ", " << D
          // << " (" << x.i()
          // << ", " << x.j()
          // << ", " << x.k()
          << " ) > m_offset: {";

        for (int i = 0; i < x.n_dimensions - 1; ++i) {
            s << x.m_offset[i] << ", ";
        }
        s << x.m_offset[x.n_dimensions - 1] << "} ]";
        return s;
    }

    /**
     * Printing type information for debug purposes
     * @param s The ostream
     * @param n/a Type selector for arg to a NON temp
     * @return ostream
     */
    template < uint_t I, typename R, typename Location, bool Temporary >
    std::ostream &operator<<(std::ostream &s, arg< I, R, Location, Temporary > const &) {
        return s << "[ arg< " << I << ", " << Temporary << " > ]";
    }
} // namespace gridtools

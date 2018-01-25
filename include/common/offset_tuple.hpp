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
#include <boost/mpl/or.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/find.hpp>
#include "defs.hpp"
#include "array.hpp"
#include "dimension.hpp"
#include "gt_assert.hpp"
#include "generic_metafunctions/binary_ops.hpp"
#include "generic_metafunctions/accumulate.hpp"
#include "generic_metafunctions/is_variadic_pack_of.hpp"
#include "generic_metafunctions/variadic_to_vector.hpp"
#include <boost/mpl/find.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/or.hpp>

namespace gridtools {

#ifdef GT_NO_CONSTEXPR_ACCESSES
#define constexpr
#endif

    /**@brief method for initializing the offsets in the placeholder
       Version valid for one dimension
       \param x is an instance of the \ref gridtools::enumtype::dimension class, which contains the offset (x.value) and
       the dimension index (X::index)
    */
    template < ushort_t N, typename X >
    GT_FUNCTION constexpr int_t initialize(X const &x) {
        GRIDTOOLS_STATIC_ASSERT(is_dimension< X >::value,
            "you passed an integer to the accessor instead of an instance of ```dimension<>```.");
        return (X::index == N ? x.value : 0);
    }

    /**@brief method for initializing the offsets in the placeholder
       Version valid for arbitrary dimension
       \param x is an instance of the \ref gridtools::dimension class, which contains the offset (x.value) and the
       dimension index (X::index)
       \param rest are the remaining arguments, which get considered one at a time in by means of recursive calls
    */
    template < ushort_t N, typename X, typename... Rest >
    GT_FUNCTION constexpr int_t initialize(X const &x, Rest const &... rest) {
        GRIDTOOLS_STATIC_ASSERT(is_dimension< X >::value,
            "you passed an integer to the accessor instead of an instance of ```dimension<>```.");
        return X::index == N ? x.value : initialize< N >(rest...);
    }

    namespace _impl {
        template < typename... GenericElements >
        struct contains_array {
            typedef typename boost::mpl::fold<
                typename variadic_to_vector< typename is_array< GenericElements >::type... >::type,
                boost::mpl::false_,
                boost::mpl::or_< boost::mpl::_1, boost::mpl::_2 > >::type type;
        };

        template < size_t ArrayDim >
        GT_FUNCTION static constexpr int_t assign_offset(uint_t pos, array< int_t, ArrayDim > const &offsets) {
            return (pos < ArrayDim) ? offsets[pos] : 0;
        }
    }

    //################################################################################
    //                              Multidimensional Fields
    //################################################################################

    /**@brief implementation of a tuple of indices (integers) that can be constexpr constructed and provde
       multifunctional API, for example setting only the indices of certain dimensions via dimension objects.
       \param n_args is the current ID of the extra dimension
       \param index_t is the index of the storage type

       EXAMPLE:

       Possible interfaces to access one extra dimension T (say, temperature) at offset -1 of a 'velocity' field V are
       the following:
       - specify it with the first integer argument (the arguments after the first define the offsets of the 3D fields
       and can be in any of the form described in gridtools::arg_type)
       \verbatim
       V(-1, x(1), z(-3));
       \endverbatim
       - specify explicitly the dimension: in this case the order of the arguments is arbitrary:
       \verbatim
       typedef dimension<4> T;
       V(x(1), z(-3), T(-1))
       \endverbatim

       Note that if no value is specified for the extra dimension a zero offset is implicitly assumed.
    */
    template < int_t Index, int_t NDim >
    struct offset_tuple : public offset_tuple< Index - 1, NDim > {
        static const int_t n_dimensions = NDim;

        typedef offset_tuple< Index - 1, NDim > super;
        static const short_t n_args = super::n_args + 1;

        /**copy constructor

           recursively assigning m_offset one by one. Works with offset_tuple of dimension lower of equal w.r.t. this
           one.
         */
        template < int_t I >
        GT_FUNCTION constexpr offset_tuple(offset_tuple< I, NDim > const &other)
            : super(other), m_offset(other.template get< n_args - 1 >()) {
            GRIDTOOLS_STATIC_ASSERT((I <= NDim), GT_INTERNAL_ERROR);
        }

        GT_FUNCTION constexpr offset_tuple(const uint_t pos, array< int_t, NDim > const &offsets)
            : super(pos + 1, offsets), m_offset(offsets[pos]) {}

        template < size_t ArrayDim,
            typename... Dimensions,
            typename Dummy = typename all_dimensions< Dimensions... >::type >
        GT_FUNCTION constexpr offset_tuple(const uint_t pos, array< int_t, ArrayDim > const &offsets, Dimensions... d)
            : super(pos + 1, offsets, d...),
              m_offset(_impl::assign_offset(pos, offsets) + initialize< super::n_dimensions - n_args >(d...)) {
            GRIDTOOLS_STATIC_ASSERT((ArrayDim <= NDim),
                GT_INTERNAL_ERROR_MSG("ERROR, can not speficy offsets with larger dimension than accessor dimensions"));
        }

        /**@brief constructor taking an integer as the first argument, and then other optional arguments.
           The integer gets assigned to the current extra dimension and the other arguments are passed to the base
           class
           (in order to get assigned to the other dimensions).
           When this constructor is used all the arguments have to be specified and passed to the function call in
           order. No check is done on the order*/
        template < typename... GenericElements,
            typename =
                typename boost::disable_if< typename _impl::contains_array< GenericElements... >::type, bool >::type >
        GT_FUNCTION constexpr offset_tuple(int const t, GenericElements const &... x)
            : super(x...), m_offset(t) {}

        /**@brief constructor taking the dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'dimension' is a
           language keyword used at the interface level.
        */
        template < ushort_t Idx,
            typename... GenericElements,
            typename Dummy = typename all_dimensions< dimension< Idx >, GenericElements... >::type >
        GT_FUNCTION constexpr offset_tuple(dimension< Idx > const &t, GenericElements const &... x)
            : super(t, x...), m_offset(initialize< super::n_dimensions - n_args + 1 >(t, x...)) {
            GRIDTOOLS_STATIC_ASSERT((Idx <= n_dimensions),
                GT_INTERNAL_ERROR_MSG("overflow in offset_tuple. Check that the accessor dimension is valid."));
        }

        // initializes recursively all the offsets to 0
        GT_FUNCTION
        constexpr offset_tuple() : super(), m_offset(0) {}

#ifdef GT_NO_CONSTEXPR_ACCESSES
#undef constexpr
#endif

        template < short_t Idx >
        GT_FUNCTION constexpr bool end() const {
            return Idx == n_args - 1 ? false : super::template end< Idx >();
        }

        /**@brief returns the offset at a specific index Idx*/
        template < short_t Idx >
        /**@brief returns the offset array*/
        GT_FUNCTION constexpr int_t get() const {
            // NOTE: this should be a constexpr whenever m_offset is a static const
            // this might not be compile-time efficient for large indexes,
            // because both taken and not taken branches are compiled. boost::mpl::eval_if would be better.
            return Idx == n_args - 1 ? m_offset : super::template get< Idx >();
        }

        /**@brief sets an element in the offset array*/
        template < short_t Idx >
        GT_FUNCTION void set(int_t offset_) {

            if (Idx == n_args - 1)
                m_offset = offset_;
            else
                super::template set< Idx >(offset_);
        }

        /**@brief sets an element in the offset array*/
        template < short_t Idx >
        GT_FUNCTION void increment(int_t offset_) {

            if (Idx == n_args - 1)
                m_offset += offset_;
            else
                super::template increment< Idx >(offset_);
        }

      protected:
        int_t m_offset;
    };

    // specialization
    template < int_t NDim >
    struct offset_tuple< 0, NDim > {
        static const int_t n_dimensions = NDim;

#ifdef GT_NO_CONSTEXPR_ACCESSES
#define constexpr
#endif

        template < size_t ArrayDim,
            typename... Dimensions,
            typename Dummy = typename all_dimensions< dimension< 0 >, Dimensions... >::type >
        GT_FUNCTION constexpr offset_tuple(const uint_t pos, array< int_t, ArrayDim > const &offsets, Dimensions... d) {
            GRIDTOOLS_STATIC_ASSERT((ArrayDim <= NDim),
                GT_INTERNAL_ERROR_MSG("ERROR, cannot specify offsets with larger dimension than accessor dimensions"));
        }

        template < typename... GenericElements,
            typename =
                typename boost::disable_if< typename _impl::contains_array< GenericElements... >::type, bool >::type >
        GT_FUNCTION constexpr offset_tuple(GenericElements const &... x) {
            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_dimension< GenericElements >::type::value...),
                GT_INTERNAL_ERROR_MSG("wrong type for the argument of an offset_tuple"));
        }

        GT_FUNCTION
        constexpr offset_tuple() {}

#ifdef GT_NO_CONSTEXPR_ACCESSES
#undef constexpr
#endif

        static const short_t n_args = 0;

        template < short_t Idx >
        GT_FUNCTION constexpr int_t get() const {
            return 0;
        }

        template < short_t Idx >
        GT_FUNCTION void set(int_t offset_) {
            // getting here is an error
        }

        template < short_t Idx >
        GT_FUNCTION void increment(int_t offset_) {
            // getting here is an error
        }
    };

    template < typename T >
    struct is_offset_tuple : boost::mpl::false_ {};

    template < int_t Index, int_t NDim >
    struct is_offset_tuple< offset_tuple< Index, NDim > > : boost::mpl::true_ {};

    template < typename T >
    class tuple_size;

    template < int_t Index, int_t NDim >
    class tuple_size< offset_tuple< Index, NDim > > : public gridtools::static_size_t< NDim > {};
} // namespace gridtools

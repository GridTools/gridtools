/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include "../stencil-composition/dimension.hpp"
#include "generic_metafunctions/logical_ops.hpp"
#include "generic_metafunctions/variadic_to_vector.hpp"
#include "generic_metafunctions/accumulate.hpp"
#include "generic_metafunctions/is_variadic_pack_of.hpp"
#include "array.hpp"

namespace gridtools {

    /**@brief method for initializing the offsets in the placeholder
       Version valid for one dimension
       \param x is an instance of the \ref gridtools::enumtype::dimension class, which contains the offset (x.value) and
       the dimension index (X::direction)
    */
    template < ushort_t N, typename X >
    GT_FUNCTION constexpr int_t initialize(X const& x) {
        GRIDTOOLS_STATIC_ASSERT(is_dimension< X >::value,
            "you passed an integer to the accessor instead of an instance of ```dimension<>```.");
        return (X::direction == N ? x.value : 0);
    }

#ifdef CXX11_ENABLED
    /**@brief method for initializing the offsets in the placeholder
       Version valid for arbitrary dimension
       \param x is an instance of the \ref gridtools::dimension class, which contains the offset (x.value) and the
       dimension index (X::direction)
       \param rest are the remaining arguments, which get considered one at a time in by means of recursive calls
    */
    template < ushort_t N, typename X, typename... Rest >
    GT_FUNCTION constexpr int_t initialize(X const& x, Rest const& ... rest) {
        GRIDTOOLS_STATIC_ASSERT(is_dimension< X >::value,
            "you passed an integer to the accessor instead of an instance of ```dimension<>```.");
        return X::direction == N ? x.value : initialize< N >(rest...);
    }
#else

    /**@brief method for initializing the offsets in the placeholder
       Version valid for two dimension
       \param x is an instance of the \ref gridtools::dimension class, which contains the offset (x.value) and the
       dimension index (X::direction)
       \param y is an instance of the \ref gridtools::dimension class, which contains the offset (y.value) and the
       dimension index (Y::direction)
    */
    template < ushort_t N, typename X, typename Y >
    GT_FUNCTION constexpr int_t initialize(X const& x, Y const& y) {
        return X::direction == N ? x.value : Y::direction == N ? y.value : 0;
    }

    /**@brief method for initializing the offsets in the placeholder
       Version valid for three dimension
       \param x is an instance of the \ref gridtools::dimension class, which contains the offset (x.value) and the
       dimension index (X::direction)
       \param y is an instance of the \ref gridtools::dimension class, which contains the offset (y.value) and the
       dimension index (Y::direction)
       \param z is an instance of the \ref gridtools::dimension class, which contains the offset (z.value) and the
       dimension index (Z::direction)
    */
    template < ushort_t N, typename X, typename Y, typename Z >
    GT_FUNCTION constexpr int_t initialize(X const& x, Y const& y, Z const& z) {
        return X::direction == N ? x.value : Y::direction == N ? y.value : Z::direction == N ? z.value : 0;
    }

    /**@brief method for initializing the offsets in the placeholder
       Version valid for three dimension
       \param x is an instance of the \ref gridtools::dimension class, which contains the offset (x.value) and the
       dimension index (X::direction)
       \param y is an instance of the \ref gridtools::dimension class, which contains the offset (y.value) and the
       dimension index (Y::direction)
       \param z is an instance of the \ref gridtools::dimension class, which contains the offset (z.value) and the
       dimension index (Z::direction)
    */
    template < ushort_t N, typename X, typename Y, typename Z, typename T >
    GT_FUNCTION constexpr int_t initialize(X const& x, Y const& y, Z const& z, T const& t) {
        return X::direction == N ? x.value : Y::direction == N ? y.value : Z::direction == N
                                                                               ? z.value
                                                                               : T::direction == N ? t.value : 0;
    }
#endif

    namespace _impl {
#ifdef CXX11_ENABLED
        template < typename... GenericElements >
        struct contains_array {
            typedef typename boost::mpl::fold<
                typename variadic_to_vector< typename is_array< GenericElements >::type... >::type,
                boost::mpl::false_,
                boost::mpl::or_< boost::mpl::_1, boost::mpl::_2 > >::type type;
        };
#endif
    }

    //################################################################################
    //                              Multidimensional Fields
    //################################################################################

    /**@brief this is a decorator of the arg_type, which is matching the extra dimensions
       \param n_args is the current ID of the extra dimension
       \param index_type is the index of the storage type

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
        static const int_t n_dim = NDim;

        typedef offset_tuple< Index - 1, NDim > super;
        static const short_t n_args = super::n_args + 1;

        /**copy constructor

           recursively assigning m_offset one by one. Works with offset_tuple of dimension lower of equal w.r.t. this one.
         */
        template<int_t I>
        GT_FUNCTION constexpr offset_tuple(offset_tuple<I, NDim> const& other) : super(other), m_offset(other.template get<n_args-1>()){
            GRIDTOOLS_STATIC_ASSERT((I <= NDim), "Internal error");
        }

        GT_FUNCTION constexpr offset_tuple(const uint_t& pos, array< int_t, NDim > const &offsets)
            : super(pos + 1, offsets), m_offset(offsets[pos]) {}
#ifdef CXX11_ENABLED

        /**@brief constructor taking an integer as the first argument, and then other optional arguments.
           The integer gets assigned to the current extra dimension and the other arguments are passed to the base
           class
           (in order to get assigned to the other dimensions).
           When this constructor is used all the arguments have to be specified and passed to the function call in
           order. No check is done on the order*/
        template < typename... GenericElements,
            typename =
                typename boost::disable_if< typename _impl::contains_array< GenericElements... >::type, bool >::type >
        GT_FUNCTION constexpr offset_tuple(int const& t, GenericElements const& ... x)
            : super(x...), m_offset(t) {}

        /**@brief constructor taking the dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'dimension' is a
           language keyword used at the interface level.
        */
        template < ushort_t Idx, typename... GenericElements >
        GT_FUNCTION constexpr offset_tuple(dimension< Idx > const& t, GenericElements const& ... x)
            : super(t, x...), m_offset(initialize< super::n_dim - n_args + 1 >(t, x...)) {
            GRIDTOOLS_STATIC_ASSERT(
                (Index <= n_dim), "overflow in offset_tuple. Check that the accessor dimension is valid.");
        }

        /**@brief constructor taking the dimension::Index class as argument.
           This allows to specify the extra arguments out of order. Note that 'dimension' is a
           language keyword used at the interface level.
        */
        template < ushort_t Idx, typename... GenericElements >
        GT_FUNCTION constexpr offset_tuple(typename dimension< Idx >::Index const& t, GenericElements const& ... x)
            : super(dimension< Idx >(0), x...),
              m_offset(initialize< super::n_dim - n_args + 1 >(dimension< Idx >(0), x...)) {
            GRIDTOOLS_STATIC_ASSERT(
                (Index <= n_dim), "overflow in offset_tuple. Check that the accessor dimension is valid.");
        }
#else
        /**@brief constructor taking an integer as the first argument, and then other optional arguments.
           The integer gets assigned to the current extra dimension and the other arguments are passed to the base
           class
           (in order to get assigned to the other dimensions).
           When this constructor is used all the arguments have to be specified and passed to the function call in
           order. No check is done on the order*/
        GT_FUNCTION offset_tuple(int const& i, int const& j, int const& k) : super(j, k), m_offset(i) {}
        GT_FUNCTION
        offset_tuple(int const& i, int const& j) : super(j), m_offset(i) {}
        GT_FUNCTION
        offset_tuple(int const& i) : m_offset(i) {}

        /**@brief constructor taking the Dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'enumtype::dimension' is a
           language keyword used at the interface level.
        */
        template < ushort_t Idx1, ushort_t Idx2, ushort_t Idx3, ushort_t Idx4 >
        GT_FUNCTION offset_tuple(
            dimension< Idx1 > const &t, dimension< Idx2 > const& u, dimension< Idx3 > const& v, dimension< Idx4 > const& h)
            : super(t, u, v, h), m_offset(initialize< super::n_dim - n_args + 1 >(t, u, v, h)) {}

        /**@brief constructor taking the Dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'enumtype::dimension' is a
           language keyword used at the interface level.
        */
        template < ushort_t Idx1, ushort_t Idx2, ushort_t Idx3 >
        GT_FUNCTION offset_tuple(dimension< Idx1 > const& t, dimension< Idx2 > const& u, dimension< Idx3 > const& v)
            : super(t, u, v), m_offset(initialize< super::n_dim - n_args + 1 >(t, u, v)) {}

        /**@brief constructor taking the Dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'dimension' is a
           language keyword used at the interface level.
        */
        template < ushort_t Idx1, ushort_t Idx2 >
        GT_FUNCTION offset_tuple(dimension< Idx1 > const& t, dimension< Idx2 > const& u)
            : super(t, u), m_offset(initialize< super::n_dim - n_args + 1 >(t, u)) {}

        /**@brief constructor taking the Dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'dimension' is a
           language keyword used at the interface level.
        */
        template < ushort_t Idx >
        GT_FUNCTION offset_tuple(dimension< Idx > const& t)
            : super(t), m_offset(initialize< super::n_dim - n_args + 1 >(t)) {}
#endif

        // initializes recursively all the offsets to 0
        GT_FUNCTION
        constexpr offset_tuple() : super(), m_offset(0) {}

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
        static const int_t n_dim = NDim;

        GT_FUNCTION constexpr offset_tuple(const uint_t pos, array< int_t, NDim > const &offsets) {}

#ifdef CXX11_ENABLED
        template < typename... GenericElements,
            typename =
                typename boost::disable_if< typename _impl::contains_array< GenericElements... >::type, bool >::type >
        GT_FUNCTION constexpr offset_tuple(GenericElements const&... x) {
            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_dimension< GenericElements >::type::value...),
                "wrong type for the argument of an offset_tuple");
        }

        // copy ctor
        GT_FUNCTION
        constexpr offset_tuple(const offset_tuple &other) {}
#else
        template < typename X, typename Y, typename Z, typename T >
        GT_FUNCTION constexpr offset_tuple(X const& x, Y const& y, Z const& z, T const& t) {}

        template < typename X, typename Y, typename Z >
        GT_FUNCTION constexpr offset_tuple(X const& x, Y const& y, Z const& z) {}

        template < typename X, typename Y >
        GT_FUNCTION constexpr offset_tuple(X const& x, Y const& y) {}

        template < typename X >
        GT_FUNCTION constexpr offset_tuple(X const& x) {}
#endif

        GT_FUNCTION
        constexpr offset_tuple() {}
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
} // namespace gridtools

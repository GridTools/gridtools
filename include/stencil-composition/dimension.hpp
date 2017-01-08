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
#include "common/host_device.hpp"

namespace gridtools {
    /**
       @section enumtype
       @{
       @brief The following struct defines one specific component of a field
       It contains a direction (compile time constant, specifying the ID of the component),
       and a value (runtime value, which is storing the offset in the given direction).
       As everything what is inside the enumtype namespace, the dimension keyword is
       supposed to be used at the application interface level.
    */
    template < ushort_t Coordinate >
    struct dimension {

        GT_FUNCTION constexpr dimension() : value(0) {}

        template < typename IntType >
        GT_FUNCTION constexpr dimension(IntType val)
            : value
#if ((!defined(CXX11_ENABLED)))
              ((uint_t)val)
#else
        {
            (uint_t) val
        }
#endif
        {
            GRIDTOOLS_STATIC_ASSERT(Coordinate != 0, "The coordinate values passed to the accessor start from 1");
            GRIDTOOLS_STATIC_ASSERT(
                Coordinate > 0, "The coordinate values passed to the accessor must be positive integerts");
        }

        /**@brief Constructor*/
        GT_FUNCTION
        constexpr dimension(dimension const &other) : value((uint_t)other.value) {}

        // TODO can I rename direction by index?
        static const ushort_t direction = Coordinate;
        static const ushort_t index = Coordinate;
        uint_t value;

        /**@brief syntactic sugar for user interface

           overloaded operators return Index types which provide the proper dimension object.
           Clarifying example:
           defining
           \code{.cpp}
           typedef dimension<5> t;
           \endcode
           we can use thefollowing alias
           \code{.cpp}
           t+2 <--> dimension<5>(2)
           \endcode

         */
    };

    template < typename T >
    struct is_dimension : boost::mpl::false_ {};

    template < ushort_t Id >
    struct is_dimension< dimension< Id > > : boost::mpl::true_ {};

#ifdef CXX11_ENABLED
    // metafunction that determines if a variadic pack are valid accessor ctr arguments
    template < typename... Types >
    struct all_dimensions {
        typedef typename boost::enable_if_c< accumulate(logical_and(), is_dimension< Types >::type::value...),
            bool >::type type;
    };
    template <>
    struct all_dimensions<> : boost::mpl::true_ {};
#endif

} // namespace gridtools

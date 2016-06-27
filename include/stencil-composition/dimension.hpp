/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
        template < typename IntType >
        GT_FUNCTION constexpr dimension(IntType val)
            : value
#if ((!defined(CXX11_ENABLED)))
              (val)
#else
        {
            val
        }
#endif
        {
            GRIDTOOLS_STATIC_ASSERT(Coordinate != 0, "The coordinate values passed to the accessor start from 1");
            GRIDTOOLS_STATIC_ASSERT(
                Coordinate > 0, "The coordinate values passed to the accessor must be positive integerts");
        }

        /**@brief Constructor*/
        GT_FUNCTION
        constexpr dimension(dimension const &other) : value(other.value) {}

        // TODO can I rename direction by index?
        static const ushort_t direction = Coordinate;
        static const ushort_t index = Coordinate;
        int_t value;

        /**@brief syntactic sugar for user interface

           overloaded operators return Index types which provide the proper dimension object.
           Clarifying example:
           defining
           \code{.cpp}
           typedef dimension<5>::Index t;
           \endcode
           we can use thefollowing alias
           \code{.cpp}
           t+2 <--> dimension<5>(2)
           \endcode

         */
        struct Index {
            GT_FUNCTION
            constexpr Index() {}
            GT_FUNCTION
            constexpr Index(Index const &) {}
            typedef dimension< Coordinate > super;
        };

      private:
        dimension();
    };

    template < typename T >
    struct is_dimension : boost::mpl::false_ {};

    template < ushort_t Id >
    struct is_dimension< dimension< Id > > : boost::mpl::true_ {};
} // namespace gridtools

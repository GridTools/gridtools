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

#include <boost/typeof/typeof.hpp>

namespace gridtools {

/*
 * HAS_TYPE_SFINAE was removed, please use the boost macro with the same functionality:
 * BOOST_MPL_HAS_XXX_TRAIT_DEF(name)
 *
 * HAS_STATIC_METHOD_SFINAE was removed as it was not used. Consider using
 * BOOST_TTI_HAS_STATIC_MEMBER_FUNCTION_GEN(name) if needed at some point.
 */

/** SFINAE method to check if a class has a method named "name" which is constexpr and returns an int*/
#define HAS_CONSTEXPR_CONSTRUCTOR(name)     \
    template <int>                          \
    struct sfinae_true : std::true_type {}; \
    template <class T>                      \
    sfinae_true<(T().name(), 0)> test(int); \
    template <class>                        \
    std::false_type test(...);              \
                                            \
    template <class T>                      \
    struct has_constexpr_name : decltype(test<T>(0)) {};

    template <int>
    struct sfinae_true : boost::mpl::true_ {};

/** SFINAE method to check if a class has a method named "name" which is constexpr and returns an int*/
#define HAS_CONSTEXPR_METHOD(instance_, name)     \
    sfinae_true<(instance_.name(), 0)> test(int); \
    template <class>                              \
    std::false_type test(...);                    \
                                                  \
    template <class T>                            \
    struct has_constexpr_name : decltype(test<T>(0)) {};

    namespace sfinae {

        /**@brief overload of the comma operator in order to use void function (the Do method)
         as arguments*/
        template <typename T>
        int operator,(T const &, int) {
            return 0;
        };

        namespace _impl {
            struct dummy_type {}; // used for SFINAE
        }                         // namespace _impl

        /**
           @brief SFINAE metafunction to detect when a static Do functor in a struct has
           2 arguments

           Used in order to make the second argument optional in the Do method of the user
           functors
        */
        template <typename Functor>
        struct has_two_args {

            static constexpr _impl::dummy_type c_ = _impl::dummy_type{};

            template <typename Derived>
            static std::false_type test(decltype(Derived::Do(c_), 0)) {
                return {};
            }

            template <typename Derived>
            static std::true_type test(decltype(Derived::Do(c_, _impl::dummy_type{}), 0)) {
                return {};
            }

            template <typename Derived>
            static std::true_type test(...) {
                return {};
            }

            typedef decltype(test<Functor>(0)) type;
            static const bool value = type::value;
        };
    } // namespace sfinae
} // namespace gridtools

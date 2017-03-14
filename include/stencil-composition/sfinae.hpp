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

#include <boost/typeof/typeof.hpp>

namespace gridtools {

    /**
       @brief Sobstitution Failure is Not An Error

       design pattern used to detect at compile-time whether a class contains a member or not (introspection)
    */
    // define an SFINAE structure
    template < typename T >
    struct SFINAE;

    template <>
    struct SFINAE< int > {};

#define HAS_TYPE_SFINAE(name, has_name, get_name)                             \
    template < typename TFunctorInternal >                                    \
    struct has_name {                                                         \
        struct MixIn {                                                        \
            using name = int;                                                 \
        };                                                                    \
        struct derived : public TFunctorInternal, public MixIn {};            \
                                                                              \
        template < typename TDerived >                                        \
        static boost::mpl::false_ test(SFINAE< typename TDerived::name > *x); \
        template < typename TDerived >                                        \
        static boost::mpl::true_ test(...);                                   \
                                                                              \
        typedef decltype(test< derived >(0)) type;                            \
        typedef TFunctorInternal functor_t;                                   \
    };                                                                        \
                                                                              \
    template < typename FunctorInternal >                                     \
    struct get_name {                                                         \
        typedef typename FunctorInternal::name type;                          \
    };

/** SFINAE method to check if a class has a method named "name" which is constexpr and returns an int*/
#define HAS_STATIC_METHOD_SFINAE(name)       \
    template < int >                         \
    struct sfinae_true : std::true_type {};  \
    template < class T >                     \
    sfinae_true< (T::name(), 0) > test(int); \
    template < class >                       \
    std::false_type test(...);               \
                                             \
    template < class T >                     \
    struct has_constexpr_name : decltype(test< T >(0)) {};

/** SFINAE method to check if a class has a method named "name" which is constexpr and returns an int*/
#define HAS_CONSTEXPR_CONSTRUCTOR(name)       \
    template < int >                          \
    struct sfinae_true : std::true_type {};   \
    template < class T >                      \
    sfinae_true< (T().name(), 0) > test(int); \
    template < class >                        \
    std::false_type test(...);                \
                                              \
    template < class T >                      \
    struct has_constexpr_name : decltype(test< T >(0)) {};

    template < int >
    struct sfinae_true : boost::mpl::true_ {};

/** SFINAE method to check if a class has a method named "name" which is constexpr and returns an int*/
#define HAS_CONSTEXPR_METHOD(instance_, name)       \
    sfinae_true< (instance_.name(), 0) > test(int); \
    template < class >                              \
    std::false_type test(...);                      \
                                                    \
    template < class T >                            \
    struct has_constexpr_name : decltype(test< T >(0)) {};

    /** @brief Implementation of introspection

        To use this define a constexpr "check" method in a class C returning and int.
        Then
        has_constexpr_check<C>
        will be either true or false wether the class has or not a default constexpr constructor.
     */
    // HAS_CONSTEXPR_CONSTRUCTOR(check)
}

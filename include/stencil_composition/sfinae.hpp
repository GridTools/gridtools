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

#ifdef CXX11_ENABLED
#define HAS_TYPE_SFINAE(name, has_name, get_name)                             \
    template < typename TFunctor >                                            \
    struct has_name {                                                         \
        struct MixIn {                                                        \
            using name = int;                                                 \
        };                                                                    \
        struct derived : public TFunctor, public MixIn {};                    \
                                                                              \
        template < typename TDerived >                                        \
        static boost::mpl::false_ test(SFINAE< typename TDerived::name > *x); \
        template < typename TDerived >                                        \
        static boost::mpl::true_ test(...);                                   \
                                                                              \
        typedef decltype(test< derived >(0)) type;                            \
        typedef TFunctor functor_t;                                           \
    };                                                                        \
                                                                              \
    template < typename Functor >                                             \
    struct get_name {                                                         \
        typedef typename Functor::name type;                                  \
    };
#else
#define HAS_TYPE_SFINAE(name, has_name, get_name)                             \
    template < typename TFunctor >                                            \
    struct has_name {                                                         \
        struct MixIn {                                                        \
            typedef int name;                                                 \
        };                                                                    \
        struct derived : public TFunctor, public MixIn {};                    \
                                                                              \
        template < typename TDerived >                                        \
        static boost::mpl::false_ test(SFINAE< typename TDerived::name > *x); \
        template < typename TDerived >                                        \
        static boost::mpl::true_ test(...);                                   \
                                                                              \
        typedef BOOST_TYPEOF(test< derived >(0)) type;                        \
        typedef TFunctor functor_t;                                           \
    };                                                                        \
                                                                              \
    template < typename Functor >                                             \
    struct get_name {                                                         \
        typedef typename Functor::name type;                                  \
    };
#endif

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

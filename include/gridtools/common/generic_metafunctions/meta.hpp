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

#include <functional>
#include <type_traits>
#include <boost/preprocessor.hpp>

#include "../defs.hpp"
#include "type_traits.hpp"
#include "gt_integer_sequence.hpp"

// internal
#define GT_META_INTERNAL_APPLY(fun, args) BOOST_PP_REMOVE_PARENS(fun)< BOOST_PP_REMOVE_PARENS(args) >

#if GT_BROKEN_TEMPLATE_ALIASES

/**
 * backward compatible way to call function
 */
#define GT_META_CALL(fun, args) typename GT_META_INTERNAL_APPLY(fun, args)::type

/**
 * backward compatible way to define an alias to the function composition
 */
#define GT_META_DEFINE_ALIAS(name, fun, args) \
    struct name : GT_META_INTERNAL_APPLY(fun, args) {}

// internal
#define GT_META_INTERNAL_LAZY_PARAM(fun) BOOST_PP_REMOVE_PARENS(fun)
#define GT_META_INTERNAL_LAZY_INLINE inline

#else

/**
 * backward compatible way to call function
 */
#define GT_META_CALL(fun, args) GT_META_INTERNAL_APPLY(fun, args)
/**
 * backward compatible way to define an alias to the function composition
 */
#define GT_META_DEFINE_ALIAS(name, fun, args) using name = GT_META_INTERNAL_APPLY(fun, args)

// internal
#define GT_META_INTERNAL_LAZY_PARAM(fun) ::gridtools::meta::force< BOOST_PP_REMOVE_PARENS(fun) >::template apply
#define GT_META_INTERNAL_LAZY_INLINE

#endif

namespace gridtools {
    /** \ingroup common
        @{
    */
    /** \ingroup allmeta
        @{
    */
    /** \defgroup meta Meta-Programming Library
        @{
    */

    /**
     *  C++11 metaprogramming library.
     *
     *  Basic Concepts
     *  ==============
     *
     *  List
     *  ----
     *  An instantiation of the template class with class template parameters.
     *
     *  Examples of lists:
     *    meta::list<void, int> : elements are void and int
     *    std::tuple<double, double> : elements are double and double
     *    std::vector<std::tuple<>, some_allocator>: elements are std::tuple<> and some_allocator
     *
     *  Examples of non lists:
     *    std::array<N, double> : first template argument is not a class
     *    int : is not the instantiation of template
     *    struct foo; is not an instantiation of template
     *
     *  Function
     *  --------
     *  A template class or an alias with class template parameters.
     *  Note the difference with MPL approach: function is not required to have `type` inner alias.
     *  Functions that have `type` inside are called lazy functions in the context of this library.
     *  The function arguments are the actual parameters of the instantiation: Arg1, Arg2 etc. in F<Arg1, Arg2 etc.>
     *  The function invocation result is just F<Arg1, Arg2 etc.> not F<Arg1, Arg2 etc.>::type.
     *  This simplification of the function concepts (comparing with MPL) is possible because of C++ aliases.
     *  And it is significant for compile time performance.
     *
     *  Examples of functions:
     *    - std::is_same
     *    - std::pair
     *    - std::tuple
     *    - meta::list
     *    - meta::is_list
     *
     *  Examples of non functions:
     *    - std::array : first parameter is not a class
     *    - meta::list<int> : is not a template
     *
     *  In the library some functions have integers as arguments. Usually they have `_c` suffix and have the sibling without
     *  prefix [This is not always the case at a moment. ]
     *  Disadvantage of having such a hybrid signature, that those functions can not be passed as arguments to high order
     *  functions.
     *
     *  Meta Class
     *  ----------
     *  A class that have `apply` inner template class or alias, which is a function [here and below the term `function`
     *  used in the context of this library]. Meta classes are used to return functions from functions.
     *
     *  Examples:
     *    - meta::always<void>
     *    - meta::rename<std::tuple>
     *
     *  High Order Function
     *  -------------------
     *  A template class or alias which first parameters are template of class class templates and the rest are classes
     *  Examples of metafuction signatures:
     *  template <template <class...> class, class...> struct foo;
     *  template <template <class...> class, template <class...> class> struct bar;
     *  template <template <class...> class...> struct baz;
     *
     *  Examples:
     *    - meta::rename
     *    - meta::lfold
     *    - meta::is_instantiation_of
     *
     *  Library Structure
     *  =================
     *
     *  It consists of the set of functions, `_c` functions and high order functions.
     *
     *  Regularly, a function has also its lazy version, which is defined in the `lazy` nested namespace under the same
     *  name. Exceptions are functions that return:
     *   - a struct with a nested `type` alias, which points to the struct itself;`
     *       ex: `list`
     *   - a struct derived from `std::intergral_constant`
     *       ex: `length`, `is_list`
     *   - meta class
     *
     *  nVidia and Intel compilers with versions < 9 and < 18 respectively have a bug that doesn't allow to use template
     *  aliases. To deal with that, the library has two modes that are switching by `GT_BROKEN_TEMPLATE_ALIASES` macro.
     *  If the value of `GT_BROKEN_TEMPLATE_ALIASES` is set to non zero, the notion of function is degradated to lazy
     *  function like in MPL.
     *
     *  In this case non-lazy functions don't exist and `lazy` nested namespace is `inline` [I.e. `meta::concat`
     *  for example is the same as `meta::lazy::concat`]. High order functions in this case interpret their functional
     *  parameters as a lazy functions [I.e. they use `::type` to invoke them].
     *
     *  `GT_META_CALL` and `GT_META_DEFINE_ALIAS` macros are defined to help keep the user code independent on that
     *  interface difference. Unfortunately in general case, it is not always possible to maintain that compatibility only
     *  using that two macros. Direct `#if GT_BROKEN_TEMPLATE_ALIASES`... could be necessary.
     *
     *  Syntax sugar: All high order functions being called with only functional arguments return partially applied versions
     *  of themselves [which became plane functions].
     *  Example, where it could be useful is:
     *  transform a list of lists:  `using out = meta::transform<meta::transform<fun>::apply, in>;`
     *
     *  Guidelines for Using Meta in Compatible with Retarded Compilers Mode
     *  =====================================================================
     *    - don't punic;
     *    - write and debug your code for some sane compiler pretending that template aliases are not a problem;
     *    - uglify each and every call of the function from meta `namespace` with `GT_META_CALL` macro;
     *    - uglify with the same macro calls to the functions that you define using composition of `meta::` functions;
     *    - replace every definition of template alias in you code with `GT_META_DEFINE_ALIAS`;
     *    - modifications above should not break compilation for the sane compiler, check it;
     *    - also check if the code compiles for your retarded compiler;
     *    - if yes, you are lucky;
     *    - if not, possible reason is that you have hand written lazy function and its `direct` counterpart that is
     *      defined smth. like `template <class T> using foo = lazy_foo<T>;` and you pass `foo` to the high order function.
     *      in this case, you need to add retarded version (where `lazy_foo` would just named `foo`) under
     *      `#if GT_BROKEN_TEMPLATE_ALIASES`;
     *    - if it is still not your case, ask @anstaf.
     *
     *  TODO List
     *  =========
     *   - rename all "hybrid" functions to `*_c` together with adding regular version.
     *   - implement cartesian product
     *   - add numeric stuff like `plus`, `less` etc.
     * *
     */
    namespace meta {

        /**
         *   The default list constructor.
         *
         *   Used within the library when it needed to produce something, that satisfy list concept.
         */
        template < class... >
        struct list {
            using type = list;
        };

        /**
         *  Remove laziness from a function
         */
        template < template < class... > class F >
        struct force {
            template < class... Args >
            using apply = typename F< Args... >::type;
        };

        /**
         *  Add laziness to a function
         */
        template < template < class... > class F >
        struct defer {
            template < class... Args >
            struct apply {
                using type = F< Args... >;
            };
        };

        /**
         *  Partially apply function F with provided arguments BoundArgs
         *
         *  Note:  if `BoundArgs...` is empty this function just converts a function to the meta class. Like mpl::quote
         */
        template < template < class... > class F, class... BoundArgs >
        struct curry {
            template < class... Args >
            GT_META_DEFINE_ALIAS(apply, F, (BoundArgs..., Args...));
        };

        /**
         *  Partially apply high order function F with provided argument G
         */
        template < template < template < class... > class, class... > class F, template < class... > class G >
        struct curry_fun {
            template < class... Args >
            GT_META_DEFINE_ALIAS(apply, F, (G, Args...));
        };

        /**
         *   list concept check.
         *
         *   Note: it is not the same as is_instance_of<list, T>.
         */
        template < class >
        struct is_list : std::false_type {};
        template < template < class... > class L, class... Ts >
        struct is_list< L< Ts... > > : std::true_type {};

        /**
         *  Check if the class has inner `type`
         */
        template < class, class = void >
        struct has_type : std::false_type {};
        template < class T >
        struct has_type< T, void_t< typename T::type > > : std::true_type {};

        /**
         *   meta class concept check
         */
        template < class, class = void >
        struct is_meta_class : std::false_type {};
        template < class T >
        struct is_meta_class< T, void_t< curry< T::template apply > > > : std::true_type {};

        template < class >
        struct length;
        template < template < class... > class L, class... Ts >
        struct length< L< Ts... > > : std::integral_constant< size_t, sizeof...(Ts) > {};

        /**
         *   Check if L is a ctor of List
         */
        template < template < class... > class L, class... Args >
        struct is_instantiation_of;
        template < template < class... > class L >
        struct is_instantiation_of< L > {
            using type = curry_fun< meta::is_instantiation_of, L >;
        };
        template < template < class... > class L, class T >
        struct is_instantiation_of< L, T > : std::false_type {};
        template < template < class... > class L, class... Ts >
        struct is_instantiation_of< L, L< Ts... > > : std::true_type {};

        /**
         *  returns predicate that is the opposite of Pred
         */
        template < template < class... > class Pred >
        struct not_ {
            template < class T >
            GT_META_DEFINE_ALIAS(apply, negation, Pred< T >);
        };

        /// placeholder definitions for bind
        template < size_t >
        struct placeholder;

        using _1 = placeholder< 0 >;
        using _2 = placeholder< 1 >;
        using _3 = placeholder< 2 >;
        using _4 = placeholder< 3 >;
        using _5 = placeholder< 4 >;
        using _6 = placeholder< 5 >;
        using _7 = placeholder< 6 >;
        using _8 = placeholder< 7 >;
        using _9 = placeholder< 8 >;
        using _10 = placeholder< 9 >;

        GT_META_INTERNAL_LAZY_INLINE namespace lazy {

            /**
             *  Normalized std::conditional version, which is proper function in the terms of meta library.
             *
             *  Note: `std::conditional` should be named `if_c` according to `meta` name convention.
             */
            template < class Cond, class Lhs, class Rhs >
            GT_META_DEFINE_ALIAS(if_, std::conditional, (Cond::value, Lhs, Rhs));

            /// some forward declarations is needed here for technical reason.
            template < class, class... >
            struct concat;
            template < class, class... >
            struct push_back;

            template < template < class... > class, class... >
            struct rename;
            template < template < class... > class, class... >
            struct lfold;
            template < template < class... > class, class... >
            struct rfold;
            template < template < class... > class, class... >
            struct combine;
            template < template < class... > class, class... >
            struct transform;
            template < template < class... > class, class... >
            struct filter;
        }

#if !GT_BROKEN_TEMPLATE_ALIASES

        /**
         *  Identity
         */
        template < class T >
        using id = T;

        template < class T >
        struct always {
            template < class... >
            using apply = T;
        };

        // 'direct' versions of lazy functions
        template < class Cond, class Lhs, class Rhs >
        using if_ = typename lazy::if_< Cond, Lhs, Rhs >::type;
        template < class List, class... Lists >
        using concat = typename lazy::concat< List, Lists... >::type;
        template < class List, class... Ts >
        using push_back = typename lazy::push_back< List, Ts... >::type;

        template < template < class... > class F, class... Args >
        using rename = typename lazy::rename< F, Args... >::type;
        template < template < class... > class F, class... Args >
        using lfold = typename lazy::lfold< F, Args... >::type;
        template < template < class... > class F, class... Args >
        using rfold = typename lazy::rfold< F, Args... >::type;
        template < template < class... > class F, class... Args >
        using combine = typename lazy::combine< F, Args... >::type;
        template < template < class... > class F, class... Args >
        using transform = typename lazy::transform< F, Args... >::type;
        template < template < class... > class F, class... Args >
        using filter = typename lazy::filter< F, Args... >::type;

#endif

        GT_META_INTERNAL_LAZY_INLINE namespace lazy {

            /**
             *  Identity (lazy)
             */
            template < class T >
            struct id {
                using type = T;
            };

            template < class T >
            struct always {
                template < class... >
                struct apply : id< T > {};
            };

            /**
             *  Extracts "producing template" from the list.
             *
             *  I.e ctor<some_instantiation_of_std_tuple>::apply is an alias of std::tuple.
             */
            template < class >
            struct ctor;
            template < template < class... > class L, class... Ts >
            struct ctor< L< Ts... > > : defer< L > {};

            template < class >
            struct clear;
            template < template < class... > class L, class... Ts >
            struct clear< L< Ts... > > {
                using type = L<>;
            };

            /**
             *   Instantiate F with the parameters taken from List.
             *
             *   Alternative interpretation: apply function F to the arguments taken form List.
             */
            template < template < class... > class To, template < class... > class From, class... Ts >
            struct rename< To, From< Ts... > > {
                using type = GT_META_CALL(To, Ts...);
            };
            template < template < class... > class To >
            struct rename< To > {
                using type = curry_fun< meta::rename, To >;
            };

            /**
             *  Convert an integer sequence to a list of corresponding integral constants.
             */
            template < class >
            struct iseq_to_list;
            template < template < class T, T... > class ISec, class Int, Int... Is >
            struct iseq_to_list< ISec< Int, Is... > > : list< std::integral_constant< Int, Is >... > {};

            /**
             *  Convert a list of integral constants to an integer sequence.
             */
            template < class >
            struct list_to_iseq;
            template < template < class... > class L, template < class T, T > class Const, class Int, Int... Is >
            struct list_to_iseq< L< Const< Int, Is >... > > {
                using type = gt_integer_sequence< Int, Is... >;
            };
            template < template < class... > class L >
            struct list_to_iseq< L<> > {
                using type = gt_index_sequence<>;
            };

            /**
             *  Make a list of integral constants of indices from 0 to N
             */
            template < size_t N >
            GT_META_DEFINE_ALIAS(make_indices, iseq_to_list, make_gt_index_sequence< N >);

            /**
             *  Make a list of integral constants of indices from 0 to length< List >
             */
            template < class List >
            GT_META_DEFINE_ALIAS(make_indices_for, make_indices, length< List >::value);

// internals
#if GT_BROKEN_TEMPLATE_ALIASES
            template < class >
            struct any_arg_impl {
                any_arg_impl(...);
            };
#else
            struct any_arg_base_impl {
                any_arg_base_impl(...);
            };
#endif
            template < class SomeList, class List >
            class drop_front_impl;
            template < class... Us, template < class... > class L, class... Ts >
            class drop_front_impl< list< Us... >, L< Ts... > > {
#if !GT_BROKEN_TEMPLATE_ALIASES
                template < class >
                using any_arg_impl = any_arg_base_impl;
#endif
                template < class... Vs >
                static L< typename Vs::type... > select(any_arg_impl< Us >..., Vs...);

              public:
                using type = decltype(select(id< Ts >()...));
            };

            /**
             *  Drop N elements from the front of the list
             *
             *  Complexity is amortized O(1).
             */
            template < class N, class List >
            GT_META_DEFINE_ALIAS(drop_front, drop_front_impl, (typename make_indices< N::value >::type, List));

            template < size_t N, class List >
            GT_META_DEFINE_ALIAS(drop_front_c, drop_front_impl, (typename make_indices< N >::type, List));

            /**
             *   Applies binary function to the elements of the list.
             *
             *   For example:
             *     combine<f>::apply<list<t1, t2, t3, t4, t5, t6, t7>> === f<f<f<t1, t2>, f<t3, f4>>, f<f<t5, t6>, t7>>
             *
             *   Complexity is amortized O(N), the depth of template instantiation is O(log(N))
             */
            template < template < class... > class F, class List, size_t N >
            struct combine_impl {
                static_assert(N > 0, "N in combine_impl<F, List, N> must be positive");
                static constexpr size_t m = N / 2;
                using type = GT_META_CALL(
                    F,
                    (typename combine_impl< F, List, m >::type,
                        typename combine_impl< F, typename drop_front_c< m, List >::type, N - m >::type));
            };
            template < template < class... > class F, template < class... > class L, class T, class... Ts >
            struct combine_impl< F, L< T, Ts... >, 1 > {
                using type = T;
            };
            template < template < class... > class F, template < class... > class L, class T1, class T2, class... Ts >
            struct combine_impl< F, L< T1, T2, Ts... >, 2 > {
                using type = GT_META_CALL(F, (T1, T2));
            };
            template < template < class... > class F,
                template < class... > class L,
                class T1,
                class T2,
                class T3,
                class... Ts >
            struct combine_impl< F, L< T1, T2, T3, Ts... >, 3 > {
                using type = GT_META_CALL(F, (T1, GT_META_CALL(F, (T2, T3))));
            };
            template < template < class... > class F,
                template < class... > class L,
                class T1,
                class T2,
                class T3,
                class T4,
                class... Ts >
            struct combine_impl< F, L< T1, T2, T3, T4, Ts... >, 4 > {
                using type = GT_META_CALL(F, (GT_META_CALL(F, (T1, T2)), GT_META_CALL(F, (T3, T4))));
            };
            template < template < class... > class F >
            struct combine< F > {
                using type = curry_fun< meta::combine, F >;
            };
            template < template < class... > class F, class List >
            struct combine< F, List > : combine_impl< F, List, length< List >::value > {};

            // internals
            template < class... Ts >
            struct inherit_impl : Ts... {};

            /**
             *   true_type if Set contains T
             *
             *   "st_" prefix stands for set
             *
             *  @pre All elements of Set are unique.
             *
             *  Complexity is O(1)
             */
            template < class Set, class T >
            struct st_contains : std::false_type {};
            template < template < class... > class L, class... Ts, class T >
            struct st_contains< L< Ts... >, T > : std::is_base_of< id< T >, inherit_impl< id< Ts >... > > {};

            /**
             *  Find the record in the map.
             *  "mp_" prefix stands for map.
             *
             *  Map is a list of lists, where the first elements of each inner lists (aka keys) are unique.
             *
             *  @return the inner list with a given Key or `void` if not found
             */
            template < class Map, class Key >
            struct mp_find;
            template < class Key, template < class... > class L, class... Ts >
            struct mp_find< L< Ts... >, Key > {
              private:
                template < template < class... > class Elem, class... Vals >
                static Elem< Key, Vals... > select(id< Elem< Key, Vals... > >);
                static void select(...);

              public:
                using type = decltype(select(std::declval< inherit_impl< id< Ts >... > >()));
            };

            template < class, class... >
            struct push_front;
            template < template < class... > class L, class... Us, class... Ts >
            struct push_front< L< Us... >, Ts... > {
                using type = L< Ts..., Us... >;
            };

            template < class, class... >
            struct push_back;
            template < template < class... > class L, class... Us, class... Ts >
            struct push_back< L< Us... >, Ts... > {
                using type = L< Us..., Ts... >;
            };

            /**
             *   Classic folds.
             *
             *   Complexity is O(N).
             *
             *   WARNING: Please use as a last resort. Consider `transform` ( which complexity is O(1) ) or `combine`
             * (which
             *   has the same complexity but O(log(N)) template depth).
             */
            template < template < class... > class F >
            struct lfold< F > {
                using type = curry_fun< meta::lfold, F >;
            };
            template < template < class... > class F, class S, template < class... > class L >
            struct lfold< F, S, L<> > {
                using type = S;
            };
            template < template < class... > class F, class S, template < class... > class L, class T >
            struct lfold< F, S, L< T > > {
                using type = GT_META_CALL(F, (S, T));
            };
            template < template < class... > class F, class S, template < class... > class L, class T1, class T2 >
            struct lfold< F, S, L< T1, T2 > > {
                using type = GT_META_CALL(F, (GT_META_CALL(F, (S, T1)), T2));
            };
            template < template < class... > class F,
                class S,
                template < class... > class L,
                class T1,
                class T2,
                class T3 >
            struct lfold< F, S, L< T1, T2, T3 > > {
                using type = GT_META_CALL(F, (GT_META_CALL(F, (GT_META_CALL(F, (S, T1)), T2)), T3));
            };
            template < template < class... > class F,
                class S,
                template < class... > class L,
                class T1,
                class T2,
                class T3,
                class T4,
                class... Ts >
            struct lfold< F, S, L< T1, T2, T3, T4, Ts... > > {
                using type = typename lfold< F,
                    GT_META_CALL(F, (GT_META_CALL(F, (GT_META_CALL(F, (GT_META_CALL(F, (S, T1)), T2)), T3)), T4)),
                    L< Ts... > >::type;
            };

            template < template < class... > class F >
            struct rfold< F > {
                using type = curry_fun< meta::rfold, F >;
            };
            template < template < class... > class F, class S, template < class... > class L >
            struct rfold< F, S, L<> > {
                using type = S;
            };
            template < template < class... > class F, class S, template < class... > class L, class T >
            struct rfold< F, S, L< T > > {
                using type = GT_META_CALL(F, (T, S));
            };
            template < template < class... > class F, class S, template < class... > class L, class T1, class T2 >
            struct rfold< F, S, L< T1, T2 > > {
                using type = GT_META_CALL(F, (T1, GT_META_CALL(F, (T2, S))));
            };
            template < template < class... > class F,
                class S,
                template < class... > class L,
                class T1,
                class T2,
                class T3 >
            struct rfold< F, S, L< T1, T2, T3 > > {
                using type = GT_META_CALL(F, (T1, GT_META_CALL(F, (T2, GT_META_CALL(F, (T3, S))))));
            };
            template < template < class... > class F,
                class S,
                template < class... > class L,
                class T1,
                class T2,
                class T3,
                class T4,
                class... Ts >
            struct rfold< F, S, L< T1, T2, T3, T4, Ts... > > {
                using type = GT_META_CALL(
                    F,
                    (T1,
                        GT_META_CALL(F,
                            (T2,
                                         GT_META_CALL(F,
                                             (T3, GT_META_CALL(F, (T4, typename rfold< F, S, L< Ts... > >::type))))))));
            };

            /**
             *   Transform `Lists` by applying `F` element wise.
             *
             *   I.e the first element of resulting list would be `F<first_from_l0, first_froml1, ...>`;
             *   the second would be `F<second_from_l0, ...>` and so on.
             *
             *   For N lists M elements each complexity is O(N). I.e for one list it is O(1).
             */
            template < template < class... > class F >
            struct transform< F > {
                using type = curry_fun< meta::transform, F >;
            };
            template < template < class... > class F, template < class... > class L, class... Ts >
            struct transform< F, L< Ts... > > {
                using type = L< GT_META_CALL(F, Ts)... >;
            };
            template < template < class... > class F,
                template < class... > class L1,
                class... T1s,
                template < class... > class L2,
                class... T2s >
            struct transform< F, L1< T1s... >, L2< T2s... > > {
                using type = L1< GT_META_CALL(F, (T1s, T2s))... >;
            };

            /**
             *  Zip lists
             */
            template < class List, class... Lists >
            GT_META_DEFINE_ALIAS(zip,
                lfold,
                (transform< meta::push_back >::type::apply, typename transform< list, List >::type, list< Lists... >));

            // transform, generic version
            template < template < class... > class F, class List, class... Lists >
            struct transform< F, List, Lists... >
                : transform< rename< F >::type::template apply, typename zip< List, Lists... >::type > {};

            /**
             *  Concatenate lists
             */
            template < template < class... > class L, class... Ts >
            struct concat< L< Ts... > > {
                using type = L< Ts... >;
            };
            template < template < class... > class L1, class... T1s, template < class... > class L2, class... T2s >
            struct concat< L1< T1s... >, L2< T2s... > > {
                using type = L1< T1s..., T2s... >;
            };

            /**
             *  Flatten a list of lists.
             */
            template < class Lists >
            GT_META_DEFINE_ALIAS(flatten, combine, (meta::concat, Lists));

            // concat, generic version
            template < class L1, class L2, class L3, class... Lists >
            struct concat< L1, L2, L3, Lists... > : flatten< list< L1, L2, L3, Lists... > > {};

            // internals
            template < template < class... > class Pred >
            struct filter_helper_impl {
                template < class T >
                GT_META_DEFINE_ALIAS(apply, meta::if_, (Pred< T >, list< T >, list<>));
            };

            /**
             *  Filter the list based of predicate
             */
            template < template < class... > class Pred >
            struct filter< Pred > {
                using type = curry_fun< meta::filter, Pred >;
            };
            template < template < class... > class Pred, class List >
            struct filter< Pred, List >
                : flatten< typename concat< list< typename clear< List >::type >,
                      typename transform< filter_helper_impl< Pred >::template apply, List >::type >::type > {};

            // internals
            template < class S, class T >
            GT_META_DEFINE_ALIAS(
                dedup_step_impl, meta::if_, (st_contains< S, T >, S, typename push_back< S, T >::type));

            /**
             *  Removes duplicates from the List.
             */
            template < class List >
            GT_META_DEFINE_ALIAS(dedup, lfold, (dedup_step_impl, typename clear< List >::type, List));

            template < class List >
            struct first;
            template < template < class... > class L, class T, class... Ts >
            struct first< L< T, Ts... > > {
                using type = T;
            };
            template < class List >
            struct second;
            template < template < class... > class L, class T, class U, class... Ts >
            struct second< L< T, U, Ts... > > {
                using type = U;
            };

            /**
             *   Take Nth element of the List
             */
            template < class List, class N >
            struct at;
            template < class List, template < class I, I > class Const, class Int >
            struct at< List, Const< Int, 0 > > : first< List > {};
            template < class List, template < class I, I > class Const, class Int >
            struct at< List, Const< Int, 1 > > : second< List > {};
            template < class List, class N >
            struct at : second< typename mp_find< typename zip< typename make_indices_for< List >::type, List >::type,
                            N >::type > {};
            template < class List, size_t N >
            GT_META_DEFINE_ALIAS(at_c, at, (List, std::integral_constant< size_t, N >));

            /**
             * return the position of T in the Set. If there is no T, it returns the length of the Set.
             *
             *  @pre All elements in Set are different.
             */
            template < class Set,
                class T,
                class Pair =
                    typename mp_find< typename zip< Set, typename make_indices_for< Set >::type >::type, T >::type >
            struct st_position : if_< std::is_void< Pair >, length< Set >, second< Pair > >::type::type {};

            /**
             *  Produce a list of N identical elements
             */
            template < size_t N, class T >
            GT_META_DEFINE_ALIAS(
                repeat, transform, (meta::always< T >::template apply, typename make_indices< N >::type));

/**
 *  NVCC bug workaround: sizeof... works incorrectly within template alias context.
 */
#ifdef __CUDACC__
            template < class... Ts >
            struct sizeof_3_dots : std::integral_constant< size_t, sizeof...(Ts) > {};

#define GT_SIZEOF_3_DOTS(Ts) ::gridtools::meta::lazy::sizeof_3_dots< Ts... >::value
#else
#define GT_SIZEOF_3_DOTS(Ts) sizeof...(Ts)
#endif

            /**
             *  C++17 drop-offs
             *
             *  Note on conjunction_fast and disjunction_fast are like std counter parts but:
             *    - short-circuiting is not implemented as required by C++17 standard
             *    - amortized complexity is O(1) because of it [in terms of the number of template instantiations].
             */
            template < class... Ts >
            GT_META_DEFINE_ALIAS(conjunction_fast,
                std::is_same,
                (list< std::integral_constant< bool, Ts::value >... >,
                                     typename repeat< GT_SIZEOF_3_DOTS(Ts), std::true_type >::type));

            template < class... Ts >
            GT_META_DEFINE_ALIAS(disjunction_fast, negation, conjunction_fast< negation< Ts >... >);

            /**
             *   all elements in lists are true
             */
            template < class List >
            struct all : rename< conjunction_fast, List >::type {};

            /**
             *   some element is true
             */
            template < class List >
            struct any : rename< disjunction_fast, List >::type {};

            /**
             *  All elements satisfy predicate
             */
            template < template < class... > class Pred, class List >
            GT_META_DEFINE_ALIAS(all_of, all, (typename transform< Pred, List >::type));

            /**
             *  Some element satisfy predicate
             */
            template < template < class... > class Pred, class List >
            GT_META_DEFINE_ALIAS(any_of, any, (typename transform< Pred, List >::type));

            template < template < class... > class Pred, template < class... > class F >
            struct selective_call_impl {
                template < class Arg >
                GT_META_DEFINE_ALIAS(apply, meta::if_, (Pred< Arg >, GT_META_CALL(F, Arg), Arg));
            };

            template < template < class... > class Pred, template < class... > class F, class List >
            GT_META_DEFINE_ALIAS(
                selective_transform, transform, (selective_call_impl< Pred, F >::template apply, List));

            /**
             *   True if the template parameter is type list which elements are all different
             */
            template < class >
            struct is_set : std::false_type {};

            template < template < class... > class L, class... Ts >
            struct is_set< L< Ts... > > : std::is_same< L< Ts... >, typename dedup< L< Ts... > >::type > {};

            /**
             *   is_set_fast evaluates to std::true_type if the parameter is a set.
             *   If parameter is not a type list, predicate evaluates to std::false_type.
             *   Compilation fails if the parameter is a type list with duplicated elements.
             *
             *   Its OK to use this predicate in static asserts and not OK in sfinae enablers.
             */
            template < class, class = void >
            struct is_set_fast : std::false_type {};

            template < template < class... > class L, class... Ts >
            struct is_set_fast< L< Ts... >, void_t< decltype(inherit_impl< id< Ts >... >{}) > > : std::true_type {};

            /**
             *   replace all Old elements to New within List
             */
            template < class List, class Old, class New >
            GT_META_DEFINE_ALIAS(replace,
                selective_transform,
                (curry< std::is_same, Old >::template apply, meta::always< New >::template apply, List));

            template < class Key >
            struct is_same_key_impl {
                template < class Elem >
                GT_META_DEFINE_ALIAS(apply, std::is_same, (Key, typename first< Elem >::type));
            };

            template < class... NewVals >
            struct replace_values_impl {
                template < class MapElem >
                struct apply;
                template < template < class... > class L, class Key, class... OldVals >
                struct apply< L< Key, OldVals... > > {
                    using type = L< Key, NewVals... >;
                };
            };

            /**
             *  replace element in the map by key
             */
            template < class Map, class Key, class... NewVals >
            GT_META_DEFINE_ALIAS(mp_replace,
                selective_transform,
                (is_same_key_impl< Key >::template apply,
                                     GT_META_INTERNAL_LAZY_PARAM(replace_values_impl< NewVals... >::template apply),
                                     Map));

            template < class N, class New >
            struct replace_at_impl {
                template < class T, class M >
                struct apply {
                    using type = T;
                };
                template < class T >
                struct apply< T, N > {
                    using type = New;
                };
            };

            /**
             *  replace element at given position
             */
            template < class List, class N, class New >
            GT_META_DEFINE_ALIAS(replace_at,
                transform,
                (GT_META_INTERNAL_LAZY_PARAM((replace_at_impl< N, New >::template apply)),
                                     List,
                                     typename make_indices_for< List >::type));
            template < class List, size_t N, class New >
            GT_META_DEFINE_ALIAS(replace_at_c, replace_at, (List, std::integral_constant< size_t, N >, New));

            template < class Arg, class... Params >
            struct replace_placeholders_impl : id< Arg > {};

            template < size_t I, class... Params >
            struct replace_placeholders_impl< placeholder< I >, Params... > : at_c< list< Params... >, I > {};
        }

        /**
         *  bind for functions
         */
        template < template < class... > class F, class... BoundArgs >
        struct bind {
            template < class... Params >
            GT_META_DEFINE_ALIAS(apply, F, (typename lazy::replace_placeholders_impl< BoundArgs, Params... >::type...));
        };

        using lazy::st_contains;
        using lazy::st_position;
        using lazy::conjunction_fast;
        using lazy::disjunction_fast;
        using lazy::all;
        using lazy::any;
        using lazy::all_of;
        using lazy::any_of;
        using lazy::is_set;
        using lazy::is_set_fast;

#if !GT_BROKEN_TEMPLATE_ALIASES

        /**
         *  Extracts "producing template" from the list.
         *
         *  I.e ctor<some_instantiation_of_std_tuple>::apply is an alias of std::tuple.
         */
        template < class >
        struct ctor;
        template < template < class... > class L, class... Ts >
        struct ctor< L< Ts... > > : curry< L > {};

        // 'direct' versions of lazy functions
        template < class List >
        using clear = typename lazy::clear< List >::type;
        template < class ISeq >
        using iseq_to_list = typename lazy::iseq_to_list< ISeq >::type;
        template < class List >
        using list_to_iseq = typename lazy::list_to_iseq< List >::type;
        template < class N, class List >
        using drop_front = typename lazy::drop_front< N, List >::type;
        template < class Map, class Key >
        using mp_find = typename lazy::mp_find< Map, Key >::type;
        template < class List, class... Ts >
        using push_front = typename lazy::push_front< List, Ts... >::type;
        template < class List >
        using first = typename lazy::first< List >::type;
        template < class List >
        using second = typename lazy::second< List >::type;
        template < class List, class N >
        using at = typename lazy::at< List, N >::type;

        template < size_t N >
        using make_indices = typename lazy::make_indices< N >::type;
        template < class List >
        using make_indices_for = typename lazy::make_indices_for< List >::type;
        template < size_t N, class List >
        using drop_front_c = typename lazy::drop_front_c< N, List >::type;
        template < class Lists >
        using flatten = typename lazy::flatten< Lists >::type;
        template < class List, class... Lists >
        using zip = typename lazy::zip< List, Lists... >::type;
        template < class List >
        using dedup = typename lazy::dedup< List >::type;
        template < class List, size_t N >
        using at_c = typename lazy::at_c< List, N >::type;
        template < size_t N, class T >
        using repeat = typename lazy::repeat< N, T >::type;
        template < template < class... > class Pred, template < class... > class F, class List >
        using selective_transform = typename lazy::selective_transform< Pred, F, List >::type;
        template < class List, class Old, class New >
        using replace = typename lazy::replace< List, Old, New >::type;
        template < class Map, class Key, class... NewVals >
        using mp_replace = typename lazy::mp_replace< Map, Key, NewVals... >::type;
        template < class List, class N, class New >
        using replace_at = typename lazy::replace_at< List, N, New >::type;
        template < class List, size_t N, class New >
        using replace_at_c = typename lazy::replace_at_c< List, N, New >::type;
#endif
    }
    /** @} */
    /** @} */
    /** @} */
}

#undef GT_META_INTERNAL_LAZY_INLINE
#undef GT_META_INTERNAL_LAZY_PARAM

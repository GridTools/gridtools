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

/**
 *  @file Minimalistic C++11 metaprogramming library.
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
 *    struct foo; is not an instantiation of template;
 *
 *  Function
 *  --------
 *  A template class or alias with class template parameters.
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
 *    - meta::_t
 *    - meta::list
 *    - meta::is_list
 *
 *  Examples of non functions:
 *    - std::array : first parameter is not a class
 *    - meta::list<int> : is not a template
 *
 *  Meta Class
 *  ----------
 *  A class that have `apply` inner class or alias, which is function.
 *  Meta classes are used to return functions from functions.
 *
 *  Examples:
 *    - meta::always<void>
 *    - meta::rename<std::tuple>
 *
 *  Meta Function
 *  -------------
 *  A template class or alias with template of class class template parameters.
 *
 *  Examples:
 *    - meta::rename
 *    - meta::lfold
 *    - meta::is_instantiation_of
 */

#include <functional>
#include <type_traits>

#include "gt_integer_sequence.hpp"

namespace gridtools {
    namespace meta {

        /**
         *  Call the function that is wrapped with the given MetaClass with the given Args
         */
        template < class MetaClass, class... Args >
        using apply = typename MetaClass::template apply< Args... >;

        /**
         *   Meta function that performs function composition.
         */
        template < template < class... > class F, template < class... > class G, template < class... > class... Fs >
        struct compose {
            template < class... Args >
            using apply = F< meta::apply< compose< G, Fs... >, Args... > >;
        };
        template < template < class... > class F, template < class... > class G >
        struct compose< F, G > {
            template < class... Args >
            using apply = F< G< Args... > >;
        };

        /**
         *  Identity function
         */
        template < class T >
        using id = T;

        /**
         *  Lazy identity function
         */
        template < class T >
        struct lazy {
            using type = T;
        };

        /**
         *  Invoke lazy function
         */
        template < class T >
        using t_ = typename T::type;

        /**
         *  Remove laziness from a function
         */
        template < template < class... > class F >
        using meta_t_ = compose< t_, F >;

        /**
         *  Remove laziness from a meta class
         */
        template < class T >
        using meta_class_t_ = meta_t_< T::template apply >;

        /**
         *  Wrap a function into a meta class
         */
        template < template < class... > class F >
        struct quote {
            template < class... Args >
            using apply = F< Args... >;
        };

        /**
         *  drop-off for C++17 void_t
         */
        template < class... >
        struct lazy_void_t {
            using type = void;
        };
        template < class... Ts >
        using void_t = t_< lazy_void_t< Ts... > >;

        /**
         *  Meta version of void_t
         */
        template < template < class... > class... >
        struct lazy_meta_void_t {
            using type = void;
        };
        template < template < class... > class... Fs >
        using meta_void_t = t_< lazy_meta_void_t< Fs... > >;

        /**
         *   The default list constructor.
         *
         *   Used within the library when it needed to produce sometning, that satisfy list concept.
         */
        template < class... >
        struct list;

        /**
         *   list concept check.
         *
         *   Note: it is not the same as is_instance<list>::apply.
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
        struct has_type< T, void_t< t_< T > > > : std::true_type {};

        /**
         *   meta class concept check
         */
        template < class, class = void >
        struct is_meta_class : std::false_type {};
        template < class T >
        struct is_meta_class< T, meta_void_t< T::template apply > > : std::true_type {};

        template < class >
        struct length;
        template < template < class... > class L, class... Ts >
        struct length< L< Ts... > > : std::integral_constant< size_t, sizeof...(Ts) > {};

        /**
         *  Extracts "producing template" from the list.
         *
         *  I.e ctor<some_instantiation_of_std_tuple>::apply is an alias of std::tuple.
         */
        template < class >
        struct ctor;
        template < template < class... > class L, class... Ts >
        struct ctor< L< Ts... > > : quote< L > {};

        template < class T >
        struct always {
            template < class... >
            using apply = T;
        };

        /**
         *   Instantiate F with the parameters taken from List.
         */
        template < template < class... > class F >
        struct lazy_rename {
            template < class >
            struct apply;
            template < template < class... > class From, class... Ts >
            struct apply< From< Ts... > > {
                using type = F< Ts... >;
            };
        };
        template < template < class... > class F >
        using rename = meta_class_t_< lazy_rename< F > >;

        /**
         *  Convert an integer sequence to a list of corresponding integral constants.
         */
        template < class >
        struct lazy_iseq_to_list;
        template < template < class T, T... > class ISec, class Int, Int... Is >
        struct lazy_iseq_to_list< ISec< Int, Is... > > {
            using type = list< std::integral_constant< Int, Is >... >;
        };
        template < class ISec >
        using iseq_to_list = t_< lazy_iseq_to_list< ISec > >;

        /**
         *  Convert a list of integral constants to an integer sequence.
         */
        template < class >
        struct lazy_list_to_iseq;
        template < template < class... > class L, template < class T, T > class Const, class Int, Int... Is >
        struct lazy_list_to_iseq< L< Const< Int, Is >... > > {
            using type = gt_integer_sequence< Int, Is... >;
        };
        template < template < class... > class L >
        struct lazy_list_to_iseq< L<> > {
            using type = gt_index_sequence<>;
        };
        template < class List >
        using list_to_iseq = t_< lazy_list_to_iseq< List > >;

        /**
         *  Make a list of integral constants of indices from 0 to N
         */
        template < size_t N >
        using make_indices = iseq_to_list< make_gt_index_sequence< N > >;

        /**
         *  Make a list of integral constants of indices from 0 to length< List >
         */
        template < class List >
        using make_indices_for = make_indices< length< List >::value >;

        // internals
        struct any_arg_impl {
            template < class T >
            any_arg_impl(T &&);
        };
        template < class SomeList, class List >
        class drop_front_impl;
        template < class... Us, template < class... > class L, class... Ts >
        class drop_front_impl< list< Us... >, L< Ts... > > {
            template < class >
            using any_arg_t = any_arg_impl;
            template < class... Vs >
            static L< t_< Vs >... > select(any_arg_t< Us >..., Vs...);

          public:
            using type = decltype(select(lazy< Ts >()...));
        };

        /**
         *  Drop N elements from the front of the list
         *
         *  Complexity is amortized O(1).
         */
        template < size_t N, class List >
        using lazy_drop_front = drop_front_impl< make_indices< N >, List >;
        template < size_t N, class List >
        using drop_front = t_< lazy_drop_front< N, List > >;

        /**
         *   Applies binary function to the elements of the list.
         *
         *   For example:
         *     combine<f>::apply<list<t1, t2, t3, t4, t5, t6, t7>> === f<f<f<t1, t2>, f<t3, f4>>, f<f<t5, t6>, t7>>
         *
         *   Complexity is amortized O(log(N))
         *
         *   If the function is associative, combine<f> has the same effect as rfold<f> and lfold<f> but faster.
         */
        template < template < class... > class F >
        struct combine {
            template < class List, size_t N >
            struct apply_impl {
                static_assert(N > 0, "N in combine_impl<F, List, N> must be positive");
                static const size_t m = N / 2;
                using type = F< t_< apply_impl< List, m > >, t_< apply_impl< drop_front< m, List >, N - m > > >;
            };
            template < template < class... > class L, class T, class... Ts >
            struct apply_impl< L< T, Ts... >, 1 > {
                using type = T;
            };
            template < template < class... > class L, class T1, class T2, class... Ts >
            struct apply_impl< L< T1, T2, Ts... >, 2 > {
                using type = F< T1, T2 >;
            };
            template < class List >
            using apply = t_< apply_impl< List, length< List >::value > >;
        };

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
        struct st_contains< L< Ts... >, T > : std::is_base_of< lazy< T >, inherit_impl< lazy< Ts >... > > {};

        /**
         *  Find the record in the map.
         *  "mp_" prefix stands for map.
         *
         *  Map is a list of lists, where the first elements of each inner lists (aka keys) are unique.
         *
         *  @return the inner list with a given Key or `void` if not found
         */
        template < class Map, class Key >
        class lazy_mp_find;
        template < class Key, template < class... > class L, class... Ts >
        class lazy_mp_find< L< Ts... >, Key > {
            template < template < class... > class Elem, class... Vals >
            static Elem< Key, Vals... > select(lazy< Elem< Key, Vals... > >);
            static void select(...);

          public:
            using type = decltype(select(std::declval< inherit_impl< lazy< Ts >... > >()));
        };
        template < class Map, class Key >
        using mp_find = t_< lazy_mp_find< Map, Key > >;

        /**
         *   Transform Lists by applying F element wise.
         *
         *   I.e the first element of resulting list would be F<first_from_l0, first_froml1, ...>;
         *   the second would be F<second_from_l0, ...> and so on.
         *
         *   For N lists M elements each complexity is O(log(N))
         */
        template < template < class... > class F >
        struct lazy_transform {
            template < class... >
            struct apply;
            template < template < class... > class L, class... Ts >
            struct apply< L< Ts... > > {
                using type = L< F< Ts >... >;
            };
            template < template < class... > class L1, class... T1s, template < class... > class L2, class... T2s >
            struct apply< L1< T1s... >, L2< T2s... > > {
                using type = L1< F< T1s, T2s >... >;
            };
            // Note, that the generic form of lazy_transform is not yet defined here.
        };
        template < template < class... > class F >
        using transform = meta_class_t_< lazy_transform< F > >;

        // internals for generic transform
        namespace transform_impl {
            // Serves as a placeholder.
            template < class... >
            struct plc;
            /// An associative binary lazy function that returns `plc` list.
            template < class T, class U >
            struct zip_helper {
                using type = plc< T, U >;
            };
            template < class T, class... Ts >
            struct zip_helper< T, plc< Ts... > > {
                using type = plc< T, Ts... >;
            };
            template < class T, class... Ts >
            struct zip_helper< plc< Ts... >, T > {
                using type = plc< Ts..., T >;
            };
            template < class... Ts, class... Us >
            struct zip_helper< plc< Ts... >, plc< Us... > > {
                using type = plc< Ts..., Us... >;
            };
        };

        // generic transform
        template < template < class... > class F >
        template < class... Lists >
        class lazy_transform< F >::apply {
            // A meta class, containing the function which takes two lists and returns the list of `plc`es from
            // the first and the second list element wise.
            // This function inherits associativity from the `zip_helper`
            using zip2 = transform< meta_t_< transform_impl::zip_helper >::apply >;
            // Now we cook general version of `zip` by applying `combine' with `zip2`.
            // It produces the list of `plc`es, collected from all argument lists element wise.
            using zip = compose< combine< zip2::apply >::apply, list >;
            // A function that renames all lists in the list to F.
            using rename_all = transform< rename< F >::template apply >;

          public:
            using type = meta::apply< rename_all, zip::apply< Lists... > >;
        };

        /**
         *   Classic folds.
         *
         *   Complexity is O(N).
         *
         *   WARNING: Please use as a last resort. Consider `transform` ( which complexity is O(1) ) or `combine` (which
         *   complexity is O(log(N))) as alternatives.
         */
        template < template < class... > class F >
        struct lazy_rfold {
            template < class, class >
            struct apply;
            template < class S, template < class... > class L >
            struct apply< S, L<> > {
                using type = S;
            };
            template < class S, template < class... > class L, class T, class... Ts >
            struct apply< S, L< T, Ts... > > {
                using type = F< T, t_< apply< S, L< Ts... > > > >;
            };
        };
        template < template < class... > class F >
        using rfold = meta_class_t_< lazy_rfold< F > >;

        template < template < class... > class F >
        struct lazy_lfold {
            template < class, class >
            struct apply;
            template < class S, template < class... > class L >
            struct apply< S, L<> > {
                using type = S;
            };
            template < class S, template < class... > class L, class T, class... Ts >
            struct apply< S, L< T, Ts... > > {
                using type = t_< apply< F< S, T >, L< Ts... > > >;
            };
        };
        template < template < class... > class F >
        using lfold = meta_class_t_< lazy_lfold< F > >;

        /**
         *  Concatenate lists
         */
        template < class... >
        struct lazy_concat;
        template < template < class... > class L, class... Ts >
        struct lazy_concat< L< Ts... > > {
            using type = L< Ts... >;
        };
        template < template < class... > class L1, class... T1s, template < class... > class L2, class... T2s >
        struct lazy_concat< L1< T1s... >, L2< T2s... > > {
            using type = L1< T1s..., T2s... >;
        };
        template < class... Lists >
        using concat = t_< lazy_concat< Lists... > >;

        /**
         *  Flatten a list of lists.
         */
        template < class Lists >
        using flatten = apply< combine< concat >, Lists >;

        template < class... Lists >
        struct lazy_concat {
            using type = flatten< list< Lists... > >;
        };

        template < class List, class... Ts >
        using push_front = concat< list< Ts... >, List >;

        template < class List, class... Ts >
        using push_back = concat< List, list< Ts... > >;

        /**
         *  Zip lists
         */
        template < class... Lists >
        using zip = apply< transform< list >, Lists... >;

        // internals
        template < template < class... > class Pred >
        struct filter_impl {
            template < class T >
            using apply = t_< std::conditional< Pred< T >::value, list< T >, list<> > >;
        };

        /**
         *  Filter the list based of predicate
         */
        template < template < class... > class Pred >
        using filter = compose< flatten, transform< filter_impl< Pred >::template apply >::template apply >;

        // internals
        template < class T, class S >
        using dedup_step_impl = t_< std::conditional< t_< st_contains< S, T > >::value, S, push_front< S, T > > >;

        /**
         *  Removes duplicates from the List
         */
        template < class List, class State = apply< ctor< List > > >
        using dedup = apply< rfold< dedup_step_impl >, State, List >;

        /**
         *   Take Nth element of the List
         */
        template < class List, class N >
        struct lazy_at;
        template < template < class... > class L, class T, class... Ts, template < class I, I > class Const, class Int >
        struct lazy_at< L< T, Ts... >, Const< Int, 0 > > {
            using type = T;
        };
        template < template < class... > class L,
            class T,
            class U,
            class... Ts,
            template < class I, I > class Const,
            class Int >
        struct lazy_at< L< T, U, Ts... >, Const< Int, 1 > > {
            using type = U;
        };
        template < class List, class N >
        using at = t_< lazy_at< List, N > >;
        template < class List, size_t N >
        using at_c = at< List, std::integral_constant< size_t, N > >;
        template < class List, size_t N >
        using lazy_at_c = lazy_at< List, std::integral_constant< size_t, N > >;
        template < class List, class N >
        struct lazy_at {
            using type = at_c< mp_find< zip< make_indices_for< List >, List >, N >, 1 >;
        };
        template < class List >
        using first = at_c< List, 0 >;
        template < class List >
        using second = at_c< List, 1 >;

        /**
         *  return the position of T in the Set
         *
         *  @pre All elements in Set are different.
         */
        template < class Set, class T, class Pair = mp_find< zip< Set, make_indices_for< Set > >, T > >
        using st_position =
            t_< t_< std::conditional< std::is_void< Pair >::value, length< Set >, lazy_at_c< Pair, 1 > > > >;

        /**
         *  Produce a list of N identical elements
         */
        template < size_t N, class T >
        using repeat = apply< transform< always< T >::template apply >, make_indices< N > >;

        /**
         *  C++17 drop-offs
         *
         *  Note on `conjunction` and `disjunction`:
         *    - short-circuiting is not implemented as required by C++17 standard
         *    - from the other side , compexity is O(1) because of it.
         */
        template < bool Val >
        using bool_constant = std::integral_constant< bool, Val >;

        template < class T >
        using negation = bool_constant< !T::value >;

        template < class... Ts >
        using conjunction =
            t_< std::is_same< list< bool_constant< Ts::value >... >, repeat< sizeof...(Ts), std::true_type > > >;

        template < class... Ts >
        using disjunction = negation< conjunction< negation< Ts >... > >;
        // end of C++17 drop-offs

        template < class List >
        using all = apply< rename< conjunction >, List >;

        template < class List >
        using any = apply< rename< disjunction >, List >;

        template < template < class... > class Pred >
        using all_of = compose< all, transform< Pred >::template apply >;

        template < template < class... > class Pred >
        using any_of = compose< any, transform< Pred >::template apply >;

        /// placeholder  definitions fo bind
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

        template < class Arg, class... Params >
        struct replace_placeholders_impl {
            using type = Arg;
        };

        template < size_t I, class... Params >
        struct replace_placeholders_impl< placeholder< I >, Params... > {
            using type = at_c< list< Params... >, I >;
        };

        /**
         *  bind for functions
         *
         *  TODO(anstaf): The signature is weird here: it is nor function nor meta function.
         *                But from the other side it is handy to use.
         *                Come up with more clean design solution
         */
        template < template < class... > class F, class... BoundArgs >
        struct bind {
            template < class... Params >
            using apply = F< t_< replace_placeholders_impl< BoundArgs, Params... > >... >;
        };

        /**
         *   Check if L is a ctor of List
         */
        template < template < class... > class L >
        struct is_instantiation_of {
            template < class List >
            struct apply : std::false_type {};
            template < class... Ts >
            struct apply< L< Ts... > > : std::true_type {};
        };
    }
}

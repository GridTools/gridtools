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
/**@file

   This files contains several helper constructs/functions in order to deal with the conditionals.
   They are used when calling @ref gridtools::make_computation
*/

#define _PAIR_CONST_REF_(count, N, data) data##Type##N const &data##Value##N

#define _PAIR_(count, N, data) data##Type##N data##Value##N

#define _APPLY_TRUE_(z, n, nil)                                                                                       \
    /**@brief recursively assigning all the conditional in the corresponding fusion vector*/                          \
    template < typename ConditionalsSet,                                                                              \
        typename First,                                                                                               \
        typename Second,                                                                                              \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType) >                                                     \
    static void apply(                                                                                                \
        ConditionalsSet &set_, First first_, Second second_, BOOST_PP_ENUM(BOOST_PP_INC(n), _PAIR_, Mss)) {           \
        boost::fusion::at_key< typename First::index_t >(set_) = first_.value();                                      \
        fill_conditionals_set< typename is_condition< typename First::first_t >::type >::apply(set_, first_.first()); \
        fill_conditionals_set< typename is_condition< typename First::second_t >::type >::apply(                      \
            set_, first_.second());                                                                                   \
        fill_conditionals_set< typename is_condition< Second >::type >::apply(                                        \
            set_, second_, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssValue));                                          \
    }

#define _APPLY_FALSE_(z, n, nil)                                                                            \
    /**@brief recursively assigning all the conditional in the corresponding fusion vector*/                \
    template < typename ConditionalsSet,                                                                    \
        typename First,                                                                                     \
        typename Second,                                                                                    \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType) >                                           \
    static void apply(                                                                                      \
        ConditionalsSet &set_, First first_, Second second_, BOOST_PP_ENUM(BOOST_PP_INC(n), _PAIR_, Mss)) { \
        fill_conditionals_set< typename is_condition< Second >::type >::apply(                              \
            set_, second_, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssValue));                                \
    }

/**@brief filling the set of conditionals with the corresponding values passed as arguments

   The arguments passed come directly from the user interface (e.g. @ref gridtools::make_mss)
 */
#define _FILL_CONDITIONALS_(z, n, nil)                                                                             \
    template < typename ConditionalsSet, typename First, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType) > \
    static void fill_conditionals(                                                                                 \
        ConditionalsSet &set_, First first_, BOOST_PP_ENUM(BOOST_PP_INC(n), _PAIR_CONST_REF_, Mss)) {              \
        fill_conditionals_set< typename boost::mpl::has_key< ConditionalsSet,                                      \
            typename if_condition_extract_index_t< First >::type >::type >::apply(set_,                            \
            first_,                                                                                                \
            BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssValue));                                                      \
        fill_conditionals(set_, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssValue));                                  \
    }

namespace gridtools {

    /**@brief simple metafunction extracting th etemplate argument*/
    template < typename T >
    struct if_condition_extract_index_t {
        typedef T type;
    };

    /**@brief specialization for @ref gridtools::condition type

       In case of @ref gridtools::condition then the type extracted is the conditional (gridtools::condition::index_t)
     */
    template < typename T1, typename T2, typename Cond >
    struct if_condition_extract_index_t< condition< T1, T2, Cond > > {
        typedef Cond type;
    };

    template < typename Conditional >
    struct fill_conditionals_set;

    /**@brief storing the conditionals passed as arguments into a container (a set)

       state machine with 2 states, determined by the template argument
       We are in one stage when the current branch of the control flow tree is itself a conditional. This is the case
       for the following specialization. If the next multi stage stencil consumed is not a conditional, we jump to
       another state,
       described by the next specialization.
     */
    template <>
    struct fill_conditionals_set< boost::mpl::true_ > {

        BOOST_PP_REPEAT(GT_MAX_MSS, _APPLY_TRUE_, _)

        /**recursion anchor*/
        template < typename ConditionalsSet, typename First >
        static void apply(ConditionalsSet &set_, First first_) {

            // if(is_conditional<First>::value)
            boost::fusion::at_key< typename First::index_t >(set_) = first_.value();

            /*binary subtree traversal*/
            fill_conditionals_set< typename is_condition< typename First::first_t >::type >::apply(
                set_, first_.first());
            fill_conditionals_set< typename is_condition< typename First::second_t >::type >::apply(
                set_, first_.second());
        }

        /**recursion anchor*/
        template < typename ConditionalsSet, typename First, typename Second >
        static void apply(ConditionalsSet &set_, First first_, Second second_) {

            // if(is_conditional<First>::value)
            boost::fusion::at_key< typename First::index_t >(set_) = first_.value();

            /*binary subtree traversal*/
            fill_conditionals_set< typename is_condition< typename First::first_t >::type >::apply(
                set_, first_.first());
            fill_conditionals_set< typename is_condition< typename First::second_t >::type >::apply(
                set_, first_.second());

            fill_conditionals_set< typename is_condition< Second >::type >::apply(set_, second_);
        }
    };

    /**@brief specialization for the case in which a branch is not a conditional

       This corresponds to another state, in the aformentioned state machine. If the subsequent multi stage stencil
       consumed is a
       conditional we jumpt to the stage above.
     */
    template <>
    struct fill_conditionals_set< boost::mpl::false_ > {

        BOOST_PP_REPEAT(GT_MAX_MSS, _APPLY_FALSE_, _)

        /**recursion anchor*/
        template < typename ConditionalsSet, typename First >
        static void apply(ConditionalsSet &set_, First first_) {}

        /**recursion anchor*/
        template < typename ConditionalsSet, typename First, typename Second >
        static void apply(ConditionalsSet &set_, First first_, Second second_) {

            fill_conditionals_set< typename is_condition< Second >::type >::apply(set_, second_);
        }
    };

    /**@brief metafunction to retrieve the @ref gridtools::conditional type from the @ref gridtools::condition*/
    template < typename Condition >
    struct condition_to_conditional {
        GRIDTOOLS_STATIC_ASSERT(is_condition< Condition >::value, "wrong type");
        typedef conditional< Condition::index_t::index_t::value > type;
    };

    /**recursion anchor*/
    template < typename ConditionalsSet, typename First >
    static void fill_conditionals(ConditionalsSet &set_, First const &first_) {
        fill_conditionals_set< typename boost::mpl::has_key< ConditionalsSet,
            typename if_condition_extract_index_t< First >::type >::type >::apply(set_, first_);
    }

    BOOST_PP_REPEAT(GT_MAX_MSS, _FILL_CONDITIONALS_, _)

    /*recursion anchor*/
    template < typename ConditionalsSet >
    static void fill_conditionals(ConditionalsSet &set_) {}

    /**@brief reaching a leaf, while recursively constructing the conditinals_set*/
    template < typename Vec, typename Mss >
    struct construct_conditionals_set {
        typedef Vec type;
    };

    /**@brief metafunction recursively traverse the binary tree to construct the conditionals_set*/
    template < typename Vec, typename Mss1, typename Mss2, typename Cond >
    struct construct_conditionals_set< Vec, condition< Mss1, Mss2, Cond > > {
        typedef typename construct_conditionals_set<
            typename construct_conditionals_set< typename boost::mpl::push_back< Vec, Cond >::type, Mss1 >::type,
            Mss2 >::type type;
    };
#undef _PAIR_CONST_REF_
#undef _PAIR_
#undef _APPLY_FALSE_
#undef _APPLY_TRUE_
#undef _FILL_CONDITIONALS_
} // namspace gridtools

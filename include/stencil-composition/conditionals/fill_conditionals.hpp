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
/**@file

   This file contains several helper constructs/functions in order to deal with the conditionals.
   They are used when calling @ref gridtools::make_computation
*/

namespace gridtools {

    /**@brief simple metafunction extracting the etemplate argument if it is not a @ref gridtools::condition*/
    template < typename T >
    struct if_condition_extract_index_t {
        typedef T type;
    };

    /**@brief specialization for @ref gridtools::condition type

       In case of @ref gridtools::condition then the type extracted is the conditional (@ref
       gridtools::condition::index_t)
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

        /**@brief recursively assigning all the conditional in the corresponding fusion vector*/
        template < typename ConditionalsSet, typename First, typename Second, typename... Mss >
        static void apply(ConditionalsSet &set_, First const &first_, Second second_, Mss const &... args_) {

            // call copy constructor
            boost::fusion::at_key< typename First::index_t >(set_) = first_.value();

            /*binary subtree traversal*/
            fill_conditionals_set< typename is_condition< typename First::first_t >::type >::apply(
                set_, first_.first());
            fill_conditionals_set< typename is_condition< typename First::second_t >::type >::apply(
                set_, first_.second());

            /*go to other branch*/
            fill_conditionals_set< typename is_condition< Second >::type >::apply(set_, second_, args_...);
        }

        /**recursion anchor*/
        template < typename ConditionalsSet, typename First >
        static void apply(ConditionalsSet &set_, First const &first_) {

            // if(is_conditional<First>::value)
            boost::fusion::at_key< typename First::index_t >(set_) = first_.value();

            /*binary subtree traversal*/
            fill_conditionals_set< typename is_condition< typename First::first_t >::type >::apply(
                set_, first_.first());
            fill_conditionals_set< typename is_condition< typename First::second_t >::type >::apply(
                set_, first_.second());
        }
    };

    /**@brief specialization for the case in which a branch is not a conditional

       This corresponds to another state, in the aformentioned state machine. If the subsequent multi stage stencil
       consumed is a
       conditional we jumpt to the stage above.
     */
    template <>
    struct fill_conditionals_set< boost::mpl::false_ > {

        /**@brief do nothing in case this branch is not a conditional, and pass to the next branch*/
        template < typename ConditionalsSet, typename First, typename Second, typename... Mss >
        static void apply(ConditionalsSet &set_, First const &first_, Second const &second_, Mss const &... args_) {
            fill_conditionals_set< typename is_condition< Second >::type >::apply(set_, second_, args_...);
        }

        /**recursion anchor*/
        template < typename ConditionalsSet, typename First >
        static void apply(ConditionalsSet &set_, First const &first_) {}
    };

    /**@brief metafunction to retrieve the @ref gridtools::conditional type from the @ref gridtools::condition*/
    template < typename Condition >
    struct condition_to_conditional {
        GRIDTOOLS_STATIC_ASSERT(is_condition< Condition >::value, "wrong type");
        typedef conditional< Condition::index_t::index_t::value > type;
    };

    /*recursion anchor*/
    template < typename ConditionalsSet >
    static void fill_conditionals(ConditionalsSet &set_) {}

    /**@brief filling the set of conditionals with the corresponding values passed as arguments

       The arguments passed come directly from the user interface (e.g. @ref gridtools::make_mss)
     */
    template < typename ConditionalsSet, typename First, typename... Mss >
    static void fill_conditionals(ConditionalsSet &set_, First const &first_, Mss const &... args_) {
        fill_conditionals_set< typename boost::mpl::has_key< ConditionalsSet,
            typename if_condition_extract_index_t< First >::type >::type >::apply(set_, first_, args_...);
        fill_conditionals(set_, args_...);
    }

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

} // namspace gridtools

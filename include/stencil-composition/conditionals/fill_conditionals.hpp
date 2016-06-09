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

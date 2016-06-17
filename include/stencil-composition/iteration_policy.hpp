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
#include "common/defs.hpp"
#include "iterate_domain.hpp"
#include "iterate_domain_metafunctions.hpp"

namespace gridtools {
    namespace _impl {

        /**\brief policy defining the behaviour on the vertical direction*/
        template < typename From, typename To, typename ZDimIndex, enumtype::execution ExecutionType >
        struct iteration_policy {};

        /**\brief specialization for the forward iteration loop over k*/
        template < typename From, typename To, typename ZDimIndex >
        struct iteration_policy< From, To, ZDimIndex, enumtype::forward > {
            static const enumtype::execution value = enumtype::forward;

            typedef From from;
            typedef To to;

            GT_FUNCTION
            static uint_t increment(uint_t &k) { return ++k; }

            template < typename IterateDomain >
            GT_FUNCTION static void increment(IterateDomain &eval) {
                GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< IterateDomain >::value), "Error: wrong type");
                eval.template increment< ZDimIndex::value, static_int< 1 > >();
            }

            GT_FUNCTION
            static bool condition(uint_t const &a, uint_t const &b) {
                return a <= b;
            } // because the k dimension excludes the extremes, so we want to loop on the internal levels (otherwise we
              // should have allocated more memory)
        };

        /**\brief specialization for the backward iteration loop over k*/
        template < typename From, typename To, typename ZDimIndex >
        struct iteration_policy< From, To, ZDimIndex, enumtype::backward > {
            static const enumtype::execution value = enumtype::backward;
            typedef To from;
            typedef From to;

            GT_FUNCTION
            static uint_t increment(uint_t &k) { return --k; }

            template < typename Domain >
            GT_FUNCTION static void increment(Domain &dom) {
                dom.template increment< ZDimIndex::value, static_int< -1 > >();
            }

            GT_FUNCTION
            static bool condition(uint_t const &a, uint_t const &b) {
                return a >= b;
            } // because the k dimension excludes the extremes, so we want to loop on the internal levels (otherwise we
              // should have allocated more memory)
        };
    } // namespace _impl
} // namespace gridtools

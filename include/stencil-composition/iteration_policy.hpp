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

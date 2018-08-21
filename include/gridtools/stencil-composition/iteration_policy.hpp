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
#include "../common/defs.hpp"
#include "./iterate_domain_fwd.hpp"

namespace gridtools {
    namespace _impl {

        /**\brief policy defining the behaviour on the vertical direction*/
        template <typename From, typename To, enumtype::execution ExecutionType>
        struct iteration_policy;

        /**\brief specialization for the forward iteration loop over k*/
        template <typename From, typename To>
        struct iteration_policy_forward {
            typedef From from;
            typedef To to;

            GT_FUNCTION
            static int_t increment(int_t &k) { return ++k; }

            template <typename IterateDomain>
            GT_FUNCTION static void increment(IterateDomain &eval) {
                GRIDTOOLS_STATIC_ASSERT(is_iterate_domain<IterateDomain>::value, "Error: wrong type");
                eval.increment_k();
            }

            GT_FUNCTION
            static bool condition(int_t const &a, int_t const &b) {
                return a <= b;
            } // because the k dimension excludes the extremes, so we want to loop on the internal levels (otherwise we
              // should have allocated more memory)
        };

        template <typename From, typename To>
        struct iteration_policy<From, To, enumtype::forward> : iteration_policy_forward<From, To> {
            static const enumtype::execution value = enumtype::forward;
        };
        template <typename From, typename To>
        struct iteration_policy<From, To, enumtype::parallel> : iteration_policy_forward<From, To> {
            static const enumtype::execution value = enumtype::parallel;
        };

        /**\brief specialization for the backward iteration loop over k*/
        template <typename From, typename To>
        struct iteration_policy<From, To, enumtype::backward> {
            static const enumtype::execution value = enumtype::backward;
            typedef To from;
            typedef From to;

            GT_FUNCTION
            static int_t increment(int_t &k) { return --k; }

            template <typename Domain>
            GT_FUNCTION static void increment(Domain &dom) {
                dom.template increment_k<-1>();
            }

            GT_FUNCTION
            static bool condition(int_t const &a, int_t const &b) {
                return a >= b;
            } // because the k dimension excludes the extremes, so we want to loop on the internal levels (otherwise we
              // should have allocated more memory)
        };

    } // namespace _impl

    template <typename T>
    struct is_iteration_policy : boost::mpl::false_ {};

    template <typename From, typename To, enumtype::execution ExecutionType>
    struct is_iteration_policy<_impl::iteration_policy<From, To, ExecutionType>> : boost::mpl::true_ {};

} // namespace gridtools

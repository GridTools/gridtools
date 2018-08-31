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

#include <type_traits>

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/meta.hpp"
#include "../common/host_device.hpp"
#include "./execution_types.hpp"
#include "./iterate_domain_fwd.hpp"
#include "./level.hpp"

namespace gridtools {

    /**\brief policy defining the behaviour on the vertical direction*/
    template <class From, class To, enumtype::execution ExecutionType>
    struct iteration_policy {
        GRIDTOOLS_STATIC_ASSERT(is_level<From>::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(is_level<To>::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(level_to_index<From>::value <= level_to_index<From>::value, GT_INTERNAL_ERROR);

        using from = From;
        using to = To;

        static constexpr enumtype::execution value = ExecutionType;

        GT_FUNCTION static int_t increment(int_t &k) { return ++k; }

        template <typename IterateDomain>
        GT_FUNCTION static void increment(IterateDomain &eval) {
            GRIDTOOLS_STATIC_ASSERT(is_iterate_domain<IterateDomain>::value, GT_INTERNAL_ERROR);
            eval.increment_k();
        }

        template <typename IterateDomain>
        GT_FUNCTION static void increment_by(IterateDomain &eval, int_t step) {
            GRIDTOOLS_STATIC_ASSERT(is_iterate_domain<IterateDomain>::value, GT_INTERNAL_ERROR);
            eval.increment_k(step);
        }

        GT_FUNCTION static bool condition(int_t a, int_t b) { return a <= b; }
    };

    /**\brief specialization for the backward iteration loop over k*/
    template <class From, class To>
    struct iteration_policy<From, To, enumtype::backward> {
        GRIDTOOLS_STATIC_ASSERT(is_level<From>::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(is_level<To>::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(level_to_index<From>::value >= level_to_index<From>::value, GT_INTERNAL_ERROR);

        using from = From;
        using to = To;

        static constexpr enumtype::execution value = enumtype::backward;

        GT_FUNCTION
        static int_t increment(int_t &k) { return --k; }

        template <typename Domain>
        GT_FUNCTION static void increment(Domain &dom) {
            GRIDTOOLS_STATIC_ASSERT(is_iterate_domain<Domain>::value, GT_INTERNAL_ERROR);
            dom.template increment_k<-1>();
        }

        template <typename IterateDomain>
        GT_FUNCTION static void increment_by(IterateDomain &eval, int_t step) {
            GRIDTOOLS_STATIC_ASSERT(is_iterate_domain<IterateDomain>::value, GT_INTERNAL_ERROR);
            eval.increment_k(-step);
        }

        GT_FUNCTION static bool condition(int_t const &a, int_t const &b) { return a >= b; }
    };

    template <typename T>
    struct is_iteration_policy : std::false_type {};

    template <typename From, typename To, enumtype::execution ExecutionType>
    struct is_iteration_policy<iteration_policy<From, To, ExecutionType>> : std::true_type {};

} // namespace gridtools

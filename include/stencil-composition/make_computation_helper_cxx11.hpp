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
#include "reductions/reduction_descriptor.hpp"

namespace gridtools {

    namespace _impl {

        /**
         * helper struct to deduce the type of a reduction and extract the initial value of a reduction passed via API.
         * specialization returns a notype when argument passed is not a reduction
         */
        template < typename >
        struct get_reduction_type {
            using type = notype;
        };

        template < typename ExecutionEngine, typename BinOp, typename EsfDescrSequence >
        struct get_reduction_type< reduction_descriptor< ExecutionEngine, BinOp, EsfDescrSequence > > {
            using type = typename reduction_descriptor< ExecutionEngine, BinOp, EsfDescrSequence >::reduction_type_t;
        };

        struct extract_reduction_intial_value_f {
            template < class T >
            notype operator()(T const &) const {
                return {};
            }
            template < typename ExecutionEngine, typename BinOp, typename EsfDescrSequence >
            auto operator()(reduction_descriptor< ExecutionEngine, BinOp, EsfDescrSequence > const &red) const
                GT_AUTO_RETURN(red.get());
        };

    } // namespace _impl
} // namespace gridtools

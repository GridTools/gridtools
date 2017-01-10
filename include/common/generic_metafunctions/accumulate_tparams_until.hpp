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
#include "../defs.hpp"

namespace gridtools {

#ifdef CXX11_ENABLED
    namespace impl {
        template < typename Value,
            typename BinaryOp,
            typename LogicalOp,
            typename First,
            typename Second,
            ushort_t Limit,
            ushort_t Cnt >
        struct accumulate_tparams_until_;

        template < typename Value,
            typename BinaryOp,
            typename LogicalOp,
            Value FirstVal,
            Value... FirstRest,
            Value SecondVal,
            Value... SecondRest,
            template < Value... > class EnvClass,
            ushort_t Limit,
            ushort_t Cnt >
        struct accumulate_tparams_until_< Value,
            BinaryOp,
            LogicalOp,
            EnvClass< FirstVal, FirstRest... >,
            EnvClass< SecondVal, SecondRest... >,
            Limit,
            Cnt > {
            static const bool value = (Cnt < Limit) ? LogicalOp()(BinaryOp()(FirstVal, SecondVal),
                                                          impl::accumulate_tparams_until_< Value,
                                                                      BinaryOp,
                                                                      LogicalOp,
                                                                      EnvClass< FirstRest... >,
                                                                      EnvClass< SecondRest... >,
                                                                      Limit,
                                                                      Cnt + 1 >::value)
                                                    : true;
        };

        template < typename Value,
            typename BinaryOp,
            typename LogicalOp,
            Value FirstVal,
            Value SecondVal,
            template < Value... > class EnvClass,
            ushort_t Limit,
            ushort_t Cnt >
        struct accumulate_tparams_until_< Value,
            BinaryOp,
            LogicalOp,
            EnvClass< FirstVal >,
            EnvClass< SecondVal >,
            Limit,
            Cnt > {
            static const bool value = (Cnt < Limit) ? BinaryOp()(FirstVal, SecondVal) : true;
        };
    }

    /*
     * @struct accumulate_tparams_until
     * will accumulate (using the LogicalOp) the result of BinaryOp over the template parameters of First and Second,
     * until a number of Limit parameters is reached. An example of use:
     * accumulate_tparams_until<int_t, equal, logical_and, extent<-1,2,-1,3,1,1>, extent<-1,2,-1,2,2,2>, 3>
     *
     * @tparam Value is the type of the template parameters being accumulated
     * @tparam BinaryOp binary operator applied to a pair of template parameters in First and Second
     * @tparam LogicalOp logical operator applied to the accumulation algorithm
     * @tparam First first operand containing a list of template parameters (subject to this algorithm)
     * @tparam Second second operand containing a list of template parameters (subject to this algorithm)
     * @tparam Limit limit number of template parameters being accumulated
     */
    template < typename Value, typename BinaryOp, typename LogicalOp, typename First, typename Second, ushort_t Limit >
    struct accumulate_tparams_until;

    template < typename Value,
        typename BinaryOp,
        typename LogicalOp,
        Value... FirstVals,
        Value... SecondVals,
        template < Value... > class EnvClass,
        ushort_t Limit >
    struct accumulate_tparams_until< Value,
        BinaryOp,
        LogicalOp,
        EnvClass< FirstVals... >,
        EnvClass< SecondVals... >,
        Limit > {
        static const bool value = impl::accumulate_tparams_until_< Value,
            BinaryOp,
            LogicalOp,
            EnvClass< FirstVals... >,
            EnvClass< SecondVals... >,
            Limit,
            0 >::value;
    };
#endif
}

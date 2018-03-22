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
#include "../defs.hpp"
#include "../host_device.hpp"
#include "../pair.hpp"

namespace gridtools {

    namespace impl {

        /**@brief accumulator recursive implementation*/
        template < typename Value, typename BinaryOp, typename LogicalOp, typename... PairVals >
        GT_FUNCTION static constexpr bool accumulate_tparams_until_(
            BinaryOp op, LogicalOp logical_op, ushort_t limit, ushort_t cnt, pair< Value, Value > pair_) {

            return (cnt < limit) ? BinaryOp()(pair_.first, pair_.second) : true;
        }

        /**@brief accumulator recursive implementation*/
        template < typename Value, typename BinaryOp, typename LogicalOp, typename... PairVals >
        GT_FUNCTION static constexpr bool accumulate_tparams_until_(BinaryOp op,
            LogicalOp logical_op,
            ushort_t limit,
            ushort_t cnt,
            pair< Value, Value > first_pair,
            PairVals... pair_vals) {
            return (cnt < limit) ? LogicalOp()(BinaryOp()(first_pair.first, first_pair.second),
                                       impl::accumulate_tparams_until_(op, logical_op, limit, cnt + 1, pair_vals...))
                                 : true;
        }
    }

    /** \ingroup common
        @{
        \ingroup allmeta
        @{
        \ingroup variadic
        @{
    */

    /*
     * @struct accumulate_tparams_until
     * will accumulate (using the LogicalOp) the result of BinaryOp over the template parameters of First and Second,
     * until a number of Limit parameters is reached. An example of use:
     * accumulate_tparams_until<equal, logical_and, extent<-1,2,-1,3,1,1>, extent<-1,2,-1,2,2,2>, 3>
     *
     * EnvClass's value type is currently restricted to int_t, due to limited compiler support of
     * template template parameters that depend on a previous template argument.
     *
     * @tparam BinaryOp binary operator applied to a pair of template parameters in First and Second
     * @tparam LogicalOp logical operator applied to the accumulation algorithm
     * @tparam First first operand containing a list of template parameters (subject to this algorithm)
     * @tparam Second second operand containing a list of template parameters (subject to this algorithm)
     * @tparam Limit limit number of template parameters being accumulated
     */
    template < typename BinaryOp, typename LogicalOp, typename First, typename Second, ushort_t Limit >
    struct accumulate_tparams_until;

    template < typename BinaryOp,
        typename LogicalOp,
        int_t... FirstVals,
        int_t... SecondVals,
        template < int_t... > class EnvClass,
        ushort_t Limit >
    struct accumulate_tparams_until< BinaryOp, LogicalOp, EnvClass< FirstVals... >, EnvClass< SecondVals... >, Limit > {
        static constexpr bool value = impl::accumulate_tparams_until_(
            BinaryOp(), LogicalOp(), Limit, 0, pair< int_t, int_t >(FirstVals, SecondVals)...);
    };
    /** @} */
    /** @} */
    /** @} */
}

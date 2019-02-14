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
#include <functional>

#include "../mss.hpp"
#include "condition.hpp"
#include "condition_tree.hpp"
/**@file*/

namespace gridtools {

    /**@brief API for specifying a boolean conditional

       \tparam Mss1 the type of the resulting multi-stage-setncil in case the condiiton is true
       \tparam Mss2 the type of the resulting multi-stage-setncil in case the condiiton is false
       \tparam Condition the type of the condition

       \param cond the runtime condition
       \param mss1_ dummy argument
       \param mss2_ dummy argument

       usage example (suppose that boolean_value is the runtime variable for the branch selection):
    @verbatim
    conditional<0>(boolean_value);

    auto comp = make_copmutation(
       if_(cond
           , make_mss(...) // true branch
           , make_mss(...) // false branch
       )
    )
    @endverbatim
    Multiple if_ statements can coexist in the same computation, and they can be arbitrarily nested
    */
    template <typename Mss1, typename Mss2, typename Condition>
    condition<Mss1, Mss2, Condition> if_(Condition cond, Mss1 const &mss1_, Mss2 const &mss2_) {
        GT_STATIC_ASSERT((std::is_convertible<Condition, std::function<bool()>>::value),
            "Condition should be nullary boolean functor.");
        GT_STATIC_ASSERT((is_condition_tree_of<Mss1, is_mss_descriptor>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_condition_tree_of<Mss2, is_mss_descriptor>::value), GT_INTERNAL_ERROR);
        return condition<Mss1, Mss2, Condition>{cond, mss1_, mss2_};
    }
} // namespace gridtools

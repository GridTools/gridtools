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

#include <assert.h>
#include <stdexcept>

#include "host_device.hpp"
#include "defs.hpp"

namespace gridtools {
    /** \ingroup common
        @{
        \defgroup error Error Handling
        @{
    */

    /**
     * @brief This struct is used to trigger runtime errors. The reason
     * for having a struct is simply that this element can be used in
     * constexpr functions while a simple call to e.g., std::runtime_error
     * would not compile.
     */
    struct error {

        template < typename T >
        GT_FUNCTION static T get(char const *msg) {
#ifdef __CUDA_ARCH__
            assert(false);
            return *((T volatile *)(0x0));
#else
            throw std::runtime_error(msg);
            assert(false);
#endif
        }

        template < typename T = uint_t >
        GT_FUNCTION static constexpr T trigger(char const *msg = "Error triggered") {
            return get< T >(msg);
        }
    };

    /**
     * @brief Helper struct used to throw an error if the condition is not met.
     * Otherwise the provided result is returned. This method can be used in constexprs.
     * @tparam T return type
     * @param cond condition that should be true
     * @param res result value
     * @param msg error message if condition is not met
     */
    template < typename T >
    GT_FUNCTION constexpr T error_or_return(bool cond, T res, char const *msg = "Error triggered") {
        return cond ? res : error::trigger< T >(msg);
    }

    /** @} */
    /** @} */
}

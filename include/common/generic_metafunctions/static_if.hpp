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

namespace gridtools {
    /** method replacing the operator ? which selects a branch at compile time and
     allows to return different types whether the condition is true or false */
    template < bool Condition >
    struct static_if;

    template <>
    struct static_if< true > {
        template < typename TrueVal, typename FalseVal >
        GT_FUNCTION static constexpr TrueVal &apply(TrueVal &true_val, FalseVal & /*false_val*/) {
            return true_val;
        }

        template < typename TrueVal, typename FalseVal >
        GT_FUNCTION static constexpr TrueVal const &apply(TrueVal const &true_val, FalseVal const & /*false_val*/) {
            return true_val;
        }

        template < typename TrueVal, typename FalseVal >
        GT_FUNCTION static void eval(TrueVal const &true_val, FalseVal const & /*false_val*/) {
            true_val();
        }
    };

    template <>
    struct static_if< false > {
        template < typename TrueVal, typename FalseVal >
        GT_FUNCTION static constexpr FalseVal &apply(TrueVal & /*true_val*/, FalseVal &false_val) {
            return false_val;
        }

        template < typename TrueVal, typename FalseVal >
        GT_FUNCTION static void eval(TrueVal const & /*true_val*/, FalseVal const &false_val) {
            false_val();
        }
    };
}

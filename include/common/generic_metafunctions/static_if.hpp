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

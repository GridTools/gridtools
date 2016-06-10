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

    template < ushort_t ID >
    struct gt_get {

        template < typename First, typename... T >
        GT_FUNCTION static constexpr const First apply(First const &first_, T const &... rest_) {
            return (ID == 0) ? first_ : gt_get< ID - 1 >::apply(rest_...);
        }

        template < typename First >
        GT_FUNCTION static constexpr const First apply(First const &first_) {
            return first_;
        }
    };
}

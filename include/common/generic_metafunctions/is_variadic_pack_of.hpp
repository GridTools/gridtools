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

#ifdef CXX11_ENABLED
    /**@brief specialization to stop the recursion*/
    template < typename... Args >
    GT_FUNCTION static constexpr bool is_variadic_pack_of(Args... args) {
        return accumulate(logical_and(), args...);
    }

#endif
} // namespace gridtools

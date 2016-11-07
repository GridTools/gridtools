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

#include <cstddef>
#include <string>

#ifdef CXX11_ENABLED

namespace gridtools {

    namespace impl_ {

        struct cstring {
            char const *ptr;
            std::size_t length;
        };

        template < typename T >
        cstring type_name_impl() {
            constexpr char const *pretty_function = __PRETTY_FUNCTION__;
            constexpr std::size_t total_size = sizeof(__PRETTY_FUNCTION__) - 1;
#ifdef __clang__
            constexpr std::size_t prefix_size =
                sizeof("gridtools::impl_::cstring gridtools::impl_::type_name_impl() [T = ") - 1;
#else // gcc
            constexpr std::size_t prefix_size =
                sizeof("gridtools::impl_::cstring gridtools::impl_::type_name_impl() [with T = ") - 1;
#endif
            constexpr std::size_t suffix_size = sizeof("]") - 1;
            return {pretty_function + prefix_size, total_size - prefix_size - suffix_size};
        }
    } // namespace impl_

    /**
     * @brief Returns a `std::string` representing the name of the given type `T`
     */
    template < typename T >
    std::string type_name() {
        auto name = impl_::type_name_impl< T >();
        return std::string(name.ptr, name.length);
    }
}

#endif // CXX11_ENABLED

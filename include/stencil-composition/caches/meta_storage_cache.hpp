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
#include "../../storage/meta_storage_base.hpp"
#include "../../common/generic_metafunctions/unzip.hpp"

namespace gridtools {

    template < typename Layout, uint_t... Dims >
    struct meta_storage_cache {

        typedef meta_storage_base< static_int< 0 >, Layout, false > meta_storage_t;

      private:
        const meta_storage_t m_value;

      public:
        GT_FUNCTION
        constexpr meta_storage_cache(meta_storage_cache const &other) : m_value{other.m_value} {};

        /**NOTE: the last 2 dimensions are Component and FD by convention*/
        GT_FUNCTION
        constexpr meta_storage_cache() : m_value{Dims...} {};

        GT_FUNCTION
        constexpr meta_storage_t value() const { return m_value; }

        GT_FUNCTION
        constexpr uint_t size() const { return m_value.size(); }

        template < ushort_t Id >
        GT_FUNCTION constexpr int_t const &strides() const {
            return m_value.template strides< Id >();
        }

        template < typename... D, typename Dummy = all_integers< typename std::remove_reference< D >::type... > >
        GT_FUNCTION constexpr int_t index(D &&... args_) const {
            return m_value.index(args_...);
        }

        template < typename Accessor >
        GT_FUNCTION constexpr int_t index(Accessor const &arg_) const {
            return m_value._index(arg_.offsets());
        }
    };
} // namespace gridtools

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

#include "id.hpp"
#include "internal/inherit.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  Find the record in the map.
         *  "mp_" prefix stands for map.
         *
         *  Map is a list of lists, where the first elements of each inner lists (aka keys) are unique.
         *
         *  @return the inner list with a given Key or `void` if not found
         */
        GT_META_LAZY_NAMESPACE {
            template <class Map, class Key>
            struct mp_find;
            template <class Key, template <class...> class L, class... Ts>
            struct mp_find<L<Ts...>, Key> {
                template <template <class...> class Elem, class... Vals>
                static Elem<Key, Vals...> select(id<Elem<Key, Vals...>> *);
                static void select(void *);

                using type = decltype(select((internal::inherit<id<Ts>...> *)0));
            };
        }
        GT_META_DELEGATE_TO_LAZY(mp_find, (class Map, class Key), (Map, Key));
    } // namespace meta
} // namespace gridtools

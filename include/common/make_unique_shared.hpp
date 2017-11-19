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

#include <memory>
#include <mutex>
#include <set>
#include <type_traits>
#include <utility>
#include <iostream>

#include "defs.hpp"

namespace gridtools {
    namespace _impl {
        namespace _make_unique_shared {
            struct ref_less {
                template < typename Ptr >
                bool operator()(const Ptr &lhs, const Ptr &rhs) const {
                    return !lhs || (rhs && *lhs < *rhs);
                }
            };

            template < typename T >
            class factory {
                std::mutex m_mutex;
                std::set< std::shared_ptr< T >, ref_less > m_set;

                void gc() {
                    for (auto i = m_set.begin(); i != m_set.end();)
                        if (i->use_count() < 2)
                            i = m_set.erase(i);
                        else
                            ++i;
                }

              public:
                template < typename... Args >
                std::shared_ptr< T > operator()(Args &&... args) {
                    std::lock_guard< std::mutex > guard(m_mutex);
                    gc();
                    return *m_set.insert(std::make_shared< T >(std::forward< Args >(args)...)).first;
                }
            };
        }
    }

    /// A factory that doesn't produce duplicate objects.
    template < typename T, typename... Args >
    std::shared_ptr< const T > make_unique_shared(Args &&... args) {
        GRIDTOOLS_STATIC_ASSERT(std::is_move_constructible< T >::value, "T should be movable.");
        static _impl::_make_unique_shared::factory< const T > factory;
        return factory(std::forward< Args >(args)...);
    }
}

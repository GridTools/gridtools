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
#include "direction.hpp"

namespace gridtools {

    struct default_predicate {
        template < typename Direction >
        bool operator()(Direction) const {
            return true;
        }
    };

    struct bitmap_predicate {
        uint_t m_boundary_bitmap; // see storage/partitioner_trivial.hpp

        enum Flag { UP = 1, LOW = 8 };

        bitmap_predicate(uint_t bm) : m_boundary_bitmap(bm) {}

        template < sign I, sign J, sign K >
        bool operator()(direction< I, J, K >) const {
            return (at_boundary(0, ((I == minus_) ? UP : LOW))) || (at_boundary(1, ((J == minus_) ? UP : LOW))) ||
                   (at_boundary(2, ((K == minus_) ? UP : LOW)));
        }

      private:
        GT_FUNCTION
        bool at_boundary(ushort_t const &component_, Flag flag_) const {
            uint_t ret = (((uint_t)flag_ * (1 << component_))) & m_boundary_bitmap;
            return ret;
        }
    };

} // namespace gridtools

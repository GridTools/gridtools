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
#include "../common/array.hpp"
#include "../common/generic_metafunctions/all_integrals.hpp"
#include "interval.hpp"
#include "level.hpp"

namespace gridtools {
    /**
     * Defines an axis_interval_t which is the former user-defined axis type and a full_interval which spans the whole
     * axis.
     * @param NIntervals Number of intervals the axis should support
     * @param ExtraOffsetsBeyondFullInterval Special case when access of k-values beyond the full_interval (i.e. the
     * last splitter value) are needed. (Note that the default interval will span the whole axis_interval_t.)
     */
    template < size_t NIntervals, int_t ExtraOffsetsBeyondFullInterval = 0 >
    class axis {
      public:
        static const uint_t max_offsets_ = cLevelOffsetLimit;

        using axis_interval_t = interval< level< 0, -1 >, level< NIntervals, 1 + ExtraOffsetsBeyondFullInterval > >;

        using full_interval = interval< level< 0, -1 >, level< NIntervals, -1 > >;

        template < typename... IntervalSizes, typename = gridtools::all_integers< IntervalSizes... > >
        axis(IntervalSizes... interval_sizes)
            : interval_sizes_{interval_sizes...} {
            GRIDTOOLS_STATIC_ASSERT(
                (sizeof...(interval_sizes) == NIntervals), "Number of intervals does not match the axis");
        }

        uint_t interval_size(const uint_t index) const { return interval_sizes_[index]; }
        const array< uint_t, NIntervals > &interval_sizes() const { return interval_sizes_; };

      private:
        array< uint_t, NIntervals > interval_sizes_;
    };
} // namespace gridtools

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

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/meta.hpp"
#include "./execution_types.hpp"
#include "./level.hpp"

namespace gridtools {
    /**
     * A helper structure that holds an information specific to the so called loop interval.
     *
     * Loop interval is limited by its From and To interval levels.
     * From level means the level from what iteration along k-axis should start. It can be upper than ToLevel
     * if the execution direction is backward.
     *
     * It is assumed that for any elementary functor within the computation at most one Do overload is used for all
     * points in this interval. In other words each elementary functor could be bound to a single interval.
     *
     * @tparam FromLevel interval level where the execution should start
     * @tparam ToLevel interval level where the execution should end
     * @tparam Payload extra compile time info
     */
    template <class FromLevel, class ToLevel, class Payload>
    struct loop_interval {
        GRIDTOOLS_STATIC_ASSERT(is_level<FromLevel>::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(is_level<ToLevel>::value, GT_INTERNAL_ERROR);

        using type = loop_interval;
    };

    template <class T>
    GT_META_DEFINE_ALIAS(is_loop_interval, meta::is_instantiation_of, (loop_interval, T));

    namespace _impl {
        GT_META_LAZY_NAMESPASE {
            template <class>
            struct reverse_loop_interval;
            template <class From, class To, class Payload>
            struct reverse_loop_interval<loop_interval<From, To, Payload>> {
                using type = loop_interval<To, From, Payload>;
            };
        }
        GT_META_DELEGATE_TO_LAZY(reverse_loop_interval, class T, T);
    } // namespace _impl

    GT_META_LAZY_NAMESPASE {
        template <class Execute, class LoopIntervals>
        struct order_loop_intervals : meta::lazy::id<LoopIntervals> {};

        template <uint_t BlockSize, class LoopIntervals>
        struct order_loop_intervals<enumtype::execute<enumtype::backward, BlockSize>, LoopIntervals> {
            using type = GT_META_CALL(
                meta::reverse, (GT_META_CALL(meta::transform, (_impl::reverse_loop_interval, LoopIntervals))));
        };
    }
    /**
     * Applies executution policy to the list of loop intervals.
     * For backward execution loop_intervals are reversed and for each interval From and To levels got swapped.
     */
    GT_META_DELEGATE_TO_LAZY(order_loop_intervals, (class Execute, class LoopIntervals), (Execute, LoopIntervals));

} // namespace gridtools

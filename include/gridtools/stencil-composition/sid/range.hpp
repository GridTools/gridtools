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

#include "../../common/array.hpp"
#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../../common/tuple.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta/at.hpp"
#include "../../meta/type_traits.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {
        namespace range_impl_ {
            template <class Ptr,
                class Strides,
                class Increments,
                class Limits,
                size_t N = tuple_util::size<Limits>::value>
            struct range {

                Ptr m_begin;
                Strides m_strides;
                Limits m_limits;

                struct iterator {
                    Ptr m_ptr;
                    range const *m_range;
                    array<ptrdiff_t, N> m_positions;

                    template <size_t I>
                    GT_FUNCTION auto limit() const GT_AUTO_RETURN(tuple_util::host_device::get<I>(m_range->m_limits));

                    template <size_t I>
                    GT_FUNCTION auto stride() const GT_AUTO_RETURN(sid::get_stride<I>(m_range->m_strides));

                    template <size_t I>
                    GT_FUNCTION auto position() const GT_AUTO_RETURN(tuple_util::host_device::get<I>(m_positions));

                    template <size_t I>
                    GT_FUNCTION auto position() GT_AUTO_RETURN(tuple_util::host_device::get<I>(m_positions));

                    template <size_t I>
                    static GT_FUNCTION GT_META_CALL(meta::at_c, (Increments, I)) increment() {
                        return {};
                    }

                    template <size_t I>
                    struct inc_f {
                        GT_FUNCTION void operator()(iterator &it) const {
                            using namespace literals;
                            if (++it.position<I>() == it.limit<I>()) {
                                sid::shift(it.m_ptr, it.stride<I>(), increment<I>() * (1_c - it.limit<I>()));
                                it.position<I>() = 0;
                                inc_f<I - 1>{};
                            } else {
                                sid::shift(it.m_ptr, it.stride<I>(), increment<I>());
                            }
                        }
                    };

                    template <>
                    struct inc_f<0> {
                        GT_FUNCTION void operator()(iterator &it) const {
                            ++it.position<0>();
                            sid::shift(it.m_ptr, it.stride<0>(), increment<0>());
                        }
                    };

                    GT_DECLARE_DEFAULT_EMPTY_CTOR(iterator);

                    GT_FUNCTION iterator(Ptr const &ptr, range const *r) : m_ptr(ptr), m_range(r) {
                        if (!tuple_util::all_of([](bool x) { return x; }, m_range->m_limits))
                            position<0>() = limit<0>();
                    }

                    GT_FUNCTION auto operator*() const GT_AUTO_RETURN(*m_ptr);

                    GT_FUNCTION void operator++() { inc_f<N - 1>(*this); }

                    GT_FUNCTION bool operator!=(iterator const &) const { return position<0>() == limit<0>(); }
                };

                GT_FUNCTION iterator begin() const { return {m_begin, this, {}}; }
                GT_FUNCTION iterator end() const { return {}; }
            };

            template <class Ptr, class Strides, class Increments, class Limits>
            struct range<Ptr, Strides, Increments, Limits, 0> {

                struct iterator {
                    Ptr const &m_ptr;

                    GT_FUNCTION auto operator*() const GT_AUTO_RETURN(*m_ptr);
                    GT_FUNCTION void operator++() {}
                    GT_FUNCTION bool operator!=(iterator const &) const { return false; }
                };

                Ptr const &m_begin;

                range(Ptr const &begin, Strides const &, Limits const &) : m_begin(begin) {}

                GT_FUNCTION Ptr const &begin() const { return {m_begin}; }
                GT_FUNCTION Ptr const &end() const { return {m_begin}; }
            };
        } // namespace range_impl_

        template <class Increments, class Ptr, class Strides, class Limits>
        GT_FUNCTION range_impl_::range<Ptr, Strides, Increments, Limits> make_range(
            Ptr const &ptr, Strides const &strides, Limits const &limits) {
            return {ptr, strides, limits};
        }
        /*
         * struct range {
         *   tuple<tuple<stride_index, step, num>>
         * }
         */
        //        make_range(ptr, strides).add_dim<0>(1_c, num).add_dim

    } // namespace sid
} // namespace gridtools

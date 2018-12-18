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

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../../common/tuple.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta/type_traits.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {
        namespace range_impl_ {
            template <class Ptr, class Strides, class Limits, size_t N = tuple_util::size<Limits>::value>
            struct range {

                Ptr m_begin;
                Strides m_strides;
                Limits m_limits;

                struct iterator {
                    Ptr m_ptr;
                    range const *m_range;
                    ptrdiff_t m_positions[N];

                    template <class Limit, enable_if_t<std::is_integral<Limit>::value, int> = 0>
                    static constexpr GT_FUNCTION ptrdiff_t inc_offset(Limit limit) {
                        return (0 < limit) - (limit < 0);
                    }

                    template <class Limit, enable_if_t<!std::is_integral<Limit>::value, int> = 0>
                    static GT_FUNCTION integral_constant<ptrdiff_t, inc_offset(Limit::value)> inc_offset(Limit) {
                        return {};
                    }

                    template <class Limit, enable_if_t<std::is_integral<Limit>::value, int> = 0>
                    static constexpr GT_FUNCTION Limit reset_offset(Limit limit) {
                        return inc_offset(limit) - limit;
                    }

                    template <class Limit, enable_if_t<!std::is_integral<Limit>::value, int> = 0>
                    static GT_FUNCTION integral_constant<typename Limit::value_type, reset_offset(Limit::value)>
                    reset_offset(Limit) {
                        return {};
                    }

                    template <size_t I>
                    GT_FUNCTION auto limit() const GT_AUTO_RETURN(tuple_util::host_device::get<I>(m_range->m_limits));

                    template <size_t I>
                    GT_FUNCTION auto stride() const GT_AUTO_RETURN(sid::get_stride<I>(m_range->m_strides));

                    template <size_t I>
                    struct reset_f {
                        GT_FUNCTION void operator()(iterator &it) const {
                            assert(it.m_positions[I] == it.limit<I>());
                            reset_f<I + 1>{}(it);
                            it.m_positions[I] = 0;
                            sid::shift(it.m_ptr, it.stride<I>(), reset_offset(it.limit<I>()));
                        }
                    };

                    template <>
                    struct reset_f<N> {
                        GT_FUNCTION void operator()(iterator &) const {}
                    };

                    template <size_t I>
                    struct inc_f {
                        GT_FUNCTION bool operator()(iterator &it) const {
                            if (inc_f<I + 1>{}(it))
                                return true;
                            ++it.m_positions[I];
                            if (it.m_positions[I] == it.limit<I>())
                                return false;
                            reset_f<I + 1>{}(it);
                            sid::shift(it.ptr, it.stride<I>(), inc_offset(it.limit<I>()));
                            return true;
                        }
                    };

                    template <>
                    struct inc_f<N> {
                        GT_FUNCTION bool operator()(iterator &) const { return false; }
                    };

                    GT_DECLARE_DEFAULT_EMPTY_CTOR(iterator);

                    GT_FUNCTION iterator(Ptr const &ptr, range const *r) : m_ptr(ptr), m_range(r), m_positions{} {
                        if (!tuple_util::all_of([](bool x) { return x; }, m_range->m_limits))
                            m_positions[0] = limit<0>();
                    }

                    GT_FUNCTION auto operator*() const GT_AUTO_RETURN(*m_ptr);

                    GT_FUNCTION void operator++() { inc_f<0>(*this); }

                    GT_FUNCTION bool operator!=(iterator const &) const { return m_positions[0] == limit<0>(); }
                };

                GT_FUNCTION iterator begin() const { return {m_begin, this, {}}; }
                GT_FUNCTION iterator end() const { return {}; }
            };

            template <class Ptr, class Strides, class Limits>
            struct range<Ptr, Strides, Limits, 0> {

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

        template <class Ptr, class Strides, class Limits>
        GT_FUNCTION range_impl_::range<Ptr, Strides, Limits> make_range(
            Ptr const &ptr, Strides const &strides, Limits const &limits) {
            return {ptr, strides, limits};
        }
    } // namespace sid
} // namespace gridtools

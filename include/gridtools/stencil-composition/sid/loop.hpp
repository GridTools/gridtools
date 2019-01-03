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

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/functional.hpp"
#include "../../common/generic_metafunctions/utility.hpp"
#include "../../common/gt_assert.hpp"
#include "../../common/host_device.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/tuple.hpp"
#include "../../meta/type_traits.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {
        namespace loop_impl_ {

            template <size_t I, class T>
            struct generic_loop {
                GRIDTOOLS_STATIC_ASSERT(std::is_signed<T>::value, GT_INTERNAL_ERROR);

                T m_num_steps;
                T m_step;

                template <class Fun>
                struct loop_f {
                    Fun m_fun;
                    T m_num_steps;
                    T m_step;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION operator()(Ptr &RESTRICT ptr, Strides const &RESTRICT strides) const {
                        assert(m_num_steps >= 0);
                        if (m_num_steps <= 0)
                            return;
                        auto &&stride = get_stride<I>(strides);
                        for (T i = 0; i < m_num_steps; ++i) {
                            m_fun(ptr, strides);
                            shift(ptr, stride, m_step);
                        }
                        shift(ptr, stride, -m_step * m_num_steps);
                    }
                };

                template <class Fun>
                constexpr GT_FUNCTION loop_f<Fun> operator()(Fun &&fun) const {
                    return {const_expr::forward<Fun>(fun), m_num_steps, m_step};
                }

                template <class Outer>
                struct cursor_f {
                    Outer m_outer;
                    T m_num_steps;
                    T m_step;
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &RESTRICT ptr, Strides const &RESTRICT strides) {
                        assert(m_num_steps >= 0);
                        if (m_num_steps <= 0)
                            return;
                        if (++m_pos == m_num_steps) {
                            shift(ptr, get_stride<I>(strides), m_step * (1 - m_num_steps));
                            m_pos = 0;
                            m_outer.next(ptr, strides);
                        } else {
                            shift(ptr, get_stride<I>(strides), m_step);
                        }
                    }

                    GT_FUNCTION bool done() const { return m_num_steps <= 0 || m_outer.done(); }
                };

                template <class Outer>
                constexpr GT_FUNCTION cursor_f<Outer> make_cursor(Outer &&outer) const {
                    return {const_expr::forward<Outer>(outer), m_num_steps, m_step, 0};
                }

                struct outer_most_cursor_f {
                    T m_step;
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &RESTRICT ptr, Strides const &RESTRICT strides) {
                        --m_pos;
                        shift(ptr, get_stride<I>(strides), m_step);
                    }

                    GT_FUNCTION bool done() const { return m_pos > 0; }
                };

                constexpr GT_FUNCTION outer_most_cursor_f make_cursor() const { return {m_step, m_num_steps}; }
            };

            template <size_t I, class T, ptrdiff_t Step>
            struct known_step_loop {
                GRIDTOOLS_STATIC_ASSERT(std::is_signed<T>::value, GT_INTERNAL_ERROR);

                T m_num_steps;

                template <class Fun>
                struct loop_f {
                    Fun m_fun;
                    T m_num_steps;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION operator()(Ptr &RESTRICT ptr, Strides const &RESTRICT strides) const {
                        assert(m_num_steps >= 0);
                        if (m_num_steps <= 0)
                            return;
                        auto &&stride = get_stride<I>(strides);
                        for (T i = 0; i < m_num_steps; ++i) {
                            m_fun(ptr, strides);
                            shift(ptr, stride, integral_constant<T, Step>{});
                        }
                        static constexpr T minus_step = -Step;
                        shift(ptr, stride, minus_step * m_num_steps);
                    }
                };

                template <class Fun>
                constexpr GT_FUNCTION loop_f<Fun> operator()(Fun &&fun) const {
                    return {const_expr::forward<Fun>(fun), m_num_steps};
                }

                template <class Outer>
                struct cursor_f {
                    Outer m_outer;
                    T m_num_steps;
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &RESTRICT ptr, Strides const &RESTRICT strides) {
                        assert(m_num_steps >= 0);
                        if (m_num_steps <= 0)
                            return;
                        if (++m_pos == m_num_steps) {
                            shift(ptr, get_stride<I>(strides), Step * (1 - m_num_steps));
                            m_pos = 0;
                            m_outer.next(ptr, strides);
                        } else {
                            shift(ptr, get_stride<I>(strides), integral_constant<T, Step>{});
                        }
                    }

                    GT_FUNCTION bool done() const { return m_num_steps <= 0 || m_outer.done(); }
                };

                template <class Outer>
                constexpr GT_FUNCTION cursor_f<Outer> make_cursor(Outer &&outer) const {
                    return {const_expr::forward<Outer>(outer), m_num_steps, 0};
                }

                struct outer_most_cursor_f {
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &RESTRICT ptr, Strides const &RESTRICT strides) {
                        --m_pos;
                        shift(ptr, get_stride<I>(strides), integral_constant<T, Step>{});
                    }

                    GT_FUNCTION bool done() const { return m_pos > 0; }
                };

                constexpr GT_FUNCTION outer_most_cursor_f make_cursor() const { return {m_num_steps}; }
            };

            template <size_t I, class T>
            struct known_step_loop<I, T, 0> {
                GRIDTOOLS_STATIC_ASSERT(std::is_signed<T>::value, GT_INTERNAL_ERROR);

                T m_num_steps;

                template <class Fun>
                struct loop_f {
                    Fun m_fun;
                    T m_num_steps;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION operator()(Ptr &RESTRICT ptr, const Strides &RESTRICT strides) const {
                        assert(m_num_steps >= 0);
                        for (T i = 0; i < m_num_steps; ++i)
                            m_fun(ptr, strides);
                    }
                };

                template <class Fun>
                constexpr GT_FUNCTION loop_f<Fun> operator()(Fun &&fun) const {
                    return {const_expr::forward<Fun>(fun), m_num_steps};
                }

                template <class Outer>
                struct cursor_f {
                    Outer m_outer;
                    T m_num_steps;
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &RESTRICT ptr, Strides const &RESTRICT strides) {
                        assert(m_num_steps >= 0);
                        if (m_num_steps <= 0)
                            return;
                        if (++m_pos == m_num_steps) {
                            m_pos = 0;
                            m_outer.next(ptr, strides);
                        }
                    }

                    GT_FUNCTION bool done() const { return m_num_steps <= 0 || m_outer.done(); }
                };

                template <class Outer>
                constexpr GT_FUNCTION cursor_f<Outer> make_cursor(Outer &&outer) const {
                    return {const_expr::forward<Outer>(outer), m_num_steps, 0};
                }

                struct outer_most_cursor_f {
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &RESTRICT ptr, Strides const &RESTRICT strides) {
                        --m_pos;
                    }

                    GT_FUNCTION bool done() const { return m_pos > 0; }
                };

                constexpr GT_FUNCTION outer_most_cursor_f make_cursor() const { return {m_num_steps}; }
            };

            template <size_t I, class T, T NumSteps>
            struct known_num_steps_loop {
                GRIDTOOLS_STATIC_ASSERT(std::is_signed<T>::value, GT_INTERNAL_ERROR);

                T m_step;

                template <class Fun>
                struct loop_f {
                    Fun m_fun;
                    T m_step;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION operator()(Ptr &RESTRICT ptr, const Strides &RESTRICT strides) const {
                        auto &&stride = get_stride<I>(strides);
                        // TODO(anstaf): to figure out if for_each<make_indices_c<NumSteps>>(...) produces better code.
                        for (T i = 0; i < NumSteps; ++i) {
                            m_fun(ptr, strides);
                            shift(ptr, stride, m_step);
                        }
                        static constexpr T minus_num_steps = -NumSteps;
                        shift(ptr, stride, m_step * minus_num_steps);
                    }
                };

                template <class Fun>
                constexpr GT_FUNCTION loop_f<Fun> operator()(Fun &&fun) const {
                    return {const_expr::forward<Fun>(fun), m_step};
                }

                template <class Outer>
                struct cursor_f {
                    Outer m_outer;
                    T m_step;
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &RESTRICT ptr, Strides const &RESTRICT strides) {
                        if (++m_pos == NumSteps) {
                            constexpr T num_steps_back = 1 - NumSteps;
                            shift(ptr, get_stride<I>(strides), m_step * num_steps_back);
                            m_pos = 0;
                            m_outer.next(ptr, strides);
                        } else {
                            shift(ptr, get_stride<I>(strides), m_step);
                        }
                    }

                    GT_FUNCTION bool done() const { return m_outer.done(); }
                };

                template <class Outer>
                constexpr GT_FUNCTION cursor_f<Outer> make_cursor(Outer &&outer) const {
                    return {const_expr::forward<Outer>(outer), m_step, 0};
                }

                struct outer_most_cursor_f {
                    T m_step;
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &RESTRICT ptr, Strides const &RESTRICT strides) {
                        --m_pos;
                        shift(ptr, get_stride<I>(strides), m_step);
                    }

                    GT_FUNCTION bool done() const { return m_pos > 0; }
                };

                constexpr GT_FUNCTION outer_most_cursor_f make_cursor() const { return {m_step, NumSteps}; }
            };

            template <size_t I, class T, ptrdiff_t NumSteps, ptrdiff_t Step>
            struct all_known_loop {
                GRIDTOOLS_STATIC_ASSERT(std::is_signed<T>::value, GT_INTERNAL_ERROR);

                template <class Fun>
                struct loop_f {
                    Fun m_fun;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION operator()(Ptr &RESTRICT ptr, const Strides &RESTRICT strides) const {
                        auto &&stride = get_stride<I>(strides);
                        for (T i = 0; i < (T)NumSteps; ++i) {
                            m_fun(ptr, strides);
                            shift(ptr, stride, integral_constant<T, Step>{});
                        }
                        shift(ptr, stride, integral_constant<T, -Step * NumSteps>{});
                    }
                };

                template <class Fun>
                constexpr GT_FUNCTION loop_f<Fun> operator()(Fun &&fun) const {
                    return {const_expr::forward<Fun>(fun)};
                }

                template <class Outer>
                struct cursor_f {
                    Outer m_outer;
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &RESTRICT ptr, Strides const &RESTRICT strides) {
                        if (++m_pos == NumSteps) {
                            constexpr T offset_back = Step * (1 - NumSteps);
                            shift(ptr, get_stride<I>(strides), offset_back);
                            m_pos = 0;
                            m_outer.next(ptr, strides);
                        } else {
                            shift(ptr, get_stride<I>(strides), Step);
                        }
                    }

                    GT_FUNCTION bool done() const { return m_outer.done(); }
                };

                template <class Outer>
                constexpr GT_FUNCTION cursor_f<Outer> make_cursor(Outer &&outer) const {
                    return {const_expr::forward<Outer>(outer), 0};
                }

                struct outer_most_cursor_f {
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &RESTRICT ptr, Strides const &RESTRICT strides) {
                        --m_pos;
                        shift(ptr, get_stride<I>(strides), Step);
                    }

                    GT_FUNCTION bool done() const { return m_pos > 0; }
                };

                constexpr GT_FUNCTION outer_most_cursor_f make_cursor() const { return {NumSteps}; }
            };

            template <size_t I, class T, ptrdiff_t NumSteps>
            struct all_known_loop<I, T, NumSteps, 0> {
                GRIDTOOLS_STATIC_ASSERT(std::is_signed<T>::value, GT_INTERNAL_ERROR);

                template <class Fun>
                struct loop_f {
                    Fun m_fun;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION operator()(Ptr &RESTRICT ptr, Strides const &RESTRICT strides) const {
                        for (T i = 0; i < (T)NumSteps; ++i)
                            m_fun(ptr, strides);
                    }
                };

                template <class Fun>
                constexpr GT_FUNCTION loop_f<Fun> operator()(Fun &&fun) const {
                    return {const_expr::forward<Fun>(fun)};
                }

                template <class Outer>
                struct cursor_f {
                    Outer m_outer;
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &RESTRICT ptr, Strides const &RESTRICT strides) {
                        if (++m_pos == NumSteps) {
                            m_pos = 0;
                            m_outer.next(ptr, strides);
                        }
                    }

                    GT_FUNCTION bool done() const { return m_outer.done(); }
                };

                template <class Outer>
                constexpr GT_FUNCTION cursor_f<Outer> make_cursor(Outer &&outer) const {
                    return {const_expr::forward<Outer>(outer), 0};
                }

                struct outer_most_cursor_f {
                    T m_pos;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &, Strides const &) {
                        --m_pos;
                    }

                    GT_FUNCTION bool done() const { return m_pos > 0; }
                };

                constexpr GT_FUNCTION outer_most_cursor_f make_cursor() const { return {NumSteps}; }
            };

            template <size_t I, class T>
            struct all_known_loop<I, T, 1, 0> {
                GRIDTOOLS_STATIC_ASSERT(std::is_signed<T>::value, GT_INTERNAL_ERROR);

                template <class Fun>
                constexpr GT_FUNCTION Fun operator()(Fun &&fun) const {
                    return fun;
                }

                template <class Outer>
                constexpr GT_FUNCTION Outer make_cursor(Outer &&outer) const {
                    return outer;
                }

                struct outer_most_cursor_f {
                    bool m_done;

                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &, Strides const &) {
                        m_done = true;
                    }

                    GT_FUNCTION bool done() const { return m_done; }
                };

                constexpr GT_FUNCTION outer_most_cursor_f make_cursor() const { return {false}; }
            };

            template <size_t I, class T>
            struct all_known_loop<I, T, 0, 0> {
                GRIDTOOLS_STATIC_ASSERT(std::is_signed<T>::value, GT_INTERNAL_ERROR);

                template <class Fun>
                constexpr GT_FUNCTION host_device::noop operator()(Fun &&) const {
                    return {};
                }

                struct cursor_f {
                    template <class Ptr, class Strides>
                    void GT_FUNCTION next(Ptr &, Strides const &) {}

                    GT_FUNCTION bool done() const { return true; }
                };

                template <class... Ts>
                constexpr GT_FUNCTION cursor_f make_cursor(Ts &&...) const {
                    return {};
                }
            };

            struct make_cursor_f {
                template <class Cursor, class Loop>
                constexpr GT_FUNCTION auto operator()(Cursor &&cursor, Loop const &loop) const
                    GT_AUTO_RETURN(loop.make_cursor(const_expr::forward<Cursor>(cursor)));
            };

            template <class Loops>
            constexpr GT_FUNCTION auto make_cursor_r(Loops &&loops)
                GT_AUTO_RETURN(tuple_util::host_device::fold(make_cursor_f{},
                    tuple_util::host_device::get<0>(const_expr::forward<Loops>(loops)).make_cursor(),
                    tuple_util::host_device::drop_front<1>(const_expr::forward<Loops>(loops))));

            template <class Loops>
            constexpr GT_FUNCTION auto make_cursor(Loops &&loops)
                GT_AUTO_RETURN(make_cursor_r(tuple_util::host_device::reverse(const_expr::forward<Loops>(loops))));

            template <class Ptr, class Strides, class Cursor>
            struct range {
                Ptr m_ptr;
                Strides const &m_strides;
                Cursor m_cursor;

                GT_FUNCTION auto operator*() const GT_AUTO_RETURN(*m_ptr);
                GT_FUNCTION void operator++() { m_cursor.next(m_ptr, m_strides); }
                template <class T>
                GT_FUNCTION bool operator!=(T &&) const {
                    return m_cursor.done();
                }

                range begin() const { return *this; }
                range end() const { return *this; }
            };

            template <class Ptr, class Strides, class Cursor>
            constexpr GT_FUNCTION range<Ptr, Strides const &, Cursor> make_range(
                Ptr ptr, Strides const &strides, Cursor &&cursor) {
                return {const_expr::move(ptr), strides, const_expr::forward<Cursor>(cursor)};
            }
        } // namespace loop_impl_

        template <size_t I,
            class T1,
            class T2,
            class T = common_type_t<T1, T2>,
            enable_if_t<std::is_integral<T1>::value && std::is_integral<T2>::value, int> = 0>
        constexpr GT_FUNCTION loop_impl_::generic_loop<I, make_signed_t<T>> make_loop(T1 num_steps, T2 step) {
            return {num_steps, step};
        }

        template <size_t I,
            class T1,
            class T2 = int,
            T2 Step = 1,
            class T = common_type_t<T1, T2>,
            enable_if_t<std::is_integral<T1>::value, int> = 0>
        constexpr GT_FUNCTION loop_impl_::known_step_loop<I, make_signed_t<T>, Step> make_loop(
            T1 num_steps, std::integral_constant<T2, Step> = {}) {
            return {num_steps};
        }

        template <size_t I,
            class T1,
            T1 NumStepsV,
            class T2,
            class T = common_type_t<T1, T2>,
            enable_if_t<std::is_integral<T1>::value && (NumStepsV > 1), int> = 0>
        constexpr GT_FUNCTION loop_impl_::known_num_steps_loop<I, make_signed_t<T>, NumStepsV> make_loop(
            std::integral_constant<T1, NumStepsV>, T2 step) {
            return {step};
        }

        template <size_t I,
            class T1,
            T1 NumStepsV,
            class T2,
            class T = common_type_t<T1, T2>,
            enable_if_t<std::is_integral<T1>::value && (NumStepsV == 0 || NumStepsV == 1), int> = 0>
        constexpr GT_FUNCTION loop_impl_::all_known_loop<I, make_signed_t<T>, NumStepsV, 0> make_loop(
            std::integral_constant<T1, NumStepsV>, T2) {
            return {};
        }

        template <size_t I,
            class T1,
            T1 NumStepsV,
            class T2 = int,
            T2 StepV = 1,
            class T = common_type_t<T1, T2>,
            enable_if_t<(NumStepsV >= 0), int> = 0>
        constexpr GT_FUNCTION loop_impl_::all_known_loop<I, make_signed_t<T>, NumStepsV, (NumStepsV > 1) ? StepV : 0>
        make_loop(std::integral_constant<T1, NumStepsV>, std::integral_constant<T2, StepV> = {}) {
            return {};
        }

        template <class Ptr, class Strides, class OuterMostLoop, class... Loops>
        constexpr GT_FUNCTION auto make_range(
            Ptr ptr, Strides const &strides, OuterMostLoop &&outer_most_loop, Loops &&... loops)
            GT_AUTO_RETURN(loop_impl_::make_range(const_expr::move(ptr),
                strides,
                loop_impl_::make_cursor(tuple<OuterMostLoop, Loops...>{
                    const_expr::forward<OuterMostLoop>(outer_most_loop), const_expr::forward<Loops>(loops)...})));

    } // namespace sid
} // namespace gridtools

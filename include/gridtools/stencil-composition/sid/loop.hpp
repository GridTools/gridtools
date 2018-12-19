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
#include "../../common/tuple_util.hpp"
#include "../../meta/type_traits.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {

        template <size_t I, class Step, class NumSteps>
        class loop;

        template <size_t I, class T>
        class loop<I, T, T> {
            GRIDTOOLS_STATIC_ASSERT(std::is_integral<T>::value, GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT(std::is_signed<T>::value, GT_INTERNAL_ERROR);

            T m_step;
            T m_num_steps;

            template <class Fun>
            struct invocation_f {
                Fun m_fun;
                T m_step;
                T m_num_steps;

                template <class Ptr, class Strides>
                void GT_FUNCTION operator()(Ptr &RESTRICT ptr, const Strides &RESTRICT strides) {
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

          public:
            constexpr GT_FUNCTION loop(T step, T num_steps) : m_step(step), m_num_steps(num_steps) {}

            template <class Fun>
            constexpr GT_FUNCTION invocation_f<decay_t<Fun>> operator()(Fun &&fun) const {
                return {const_expr::forward<Fun>(fun), m_step, m_num_steps};
            }

            static constexpr size_t index = I;

            constexpr GT_FUNCTION T step() const { return m_step; }
            constexpr GT_FUNCTION T num_steps() const { return m_num_steps; }
        };

        template <size_t I, class T, T Step>
        class loop<I, integral_constant<T, Step>, T> {
            GRIDTOOLS_STATIC_ASSERT(std::is_integral<T>::value, GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT(std::is_signed<T>::value, GT_INTERNAL_ERROR);

            T m_num_steps;

            template <class Fun>
            struct invocation_f {
                Fun m_fun;
                T m_num_steps;

                template <class Ptr, class Strides>
                void GT_FUNCTION operator()(Ptr &RESTRICT ptr, const Strides &RESTRICT strides) {
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

          public:
            template <class U>
            constexpr GT_FUNCTION loop(U &&, T num_steps) : m_num_steps(num_steps) {}

            template <class Fun>
            constexpr GT_FUNCTION invocation_f<decay_t<Fun>> operator()(Fun &&fun) const {
                return {const_expr::forward<Fun>(fun), m_num_steps};
            }

            static constexpr size_t index = I;

            constexpr GT_FUNCTION integral_constant<T, Step> step() const { return {}; }
            constexpr GT_FUNCTION T num_steps() const { return m_num_steps; }
        };

        template <size_t I, class T>
        class loop<I, integral_constant<T, 0>, T> {
            GRIDTOOLS_STATIC_ASSERT(std::is_integral<T>::value, GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT(std::is_signed<T>::value, GT_INTERNAL_ERROR);

            T m_num_steps;

            template <class Fun>
            struct invocation_f {
                Fun m_fun;
                T m_num_steps;

                template <class Ptr, class Strides>
                void GT_FUNCTION operator()(Ptr &RESTRICT ptr, const Strides &RESTRICT strides) {
                    assert(m_num_steps >= 0);
                    for (T i = 0; i < m_num_steps; ++i)
                        m_fun(ptr, strides);
                }
            };

          public:
            template <class U>
            constexpr GT_FUNCTION loop(U &&, T num_steps) : m_num_steps(num_steps) {}

            template <class Fun>
            constexpr GT_FUNCTION invocation_f<decay_t<Fun>> operator()(Fun &&fun) const {
                return {const_expr::forward<Fun>(fun), m_num_steps};
            }

            static constexpr size_t index = I;

            constexpr GT_FUNCTION integral_constant<T, 0> step() const { return {}; }
            constexpr GT_FUNCTION T num_steps() const { return m_num_steps; }
        };

        template <size_t I, class T, T NumSteps>
        class loop<I, T, integral_constant<T, NumSteps>> {
            GRIDTOOLS_STATIC_ASSERT(std::is_integral<T>::value, GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT(std::is_signed<T>::value, GT_INTERNAL_ERROR);

            T m_step;

            template <class Fun>
            struct invocation_f {
                Fun m_fun;
                T m_step;

                template <class Ptr, class Strides>
                void GT_FUNCTION operator()(Ptr &RESTRICT ptr, const Strides &RESTRICT strides) {
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

          public:
            template <class U>
            constexpr GT_FUNCTION loop(T step, U &&) : m_step(step) {}

            template <class Fun>
            constexpr GT_FUNCTION invocation_f<decay_t<Fun>> operator()(Fun &&fun) const {
                return {const_expr::forward<Fun>(fun), m_step};
            }

            static constexpr size_t index = I;

            constexpr GT_FUNCTION T step() const { return m_step; }
            constexpr GT_FUNCTION integral_constant<T, NumSteps> num_steps() const { return {}; }
        };

        template <size_t I, class T, T Step, T NumSteps>
        class loop<I, integral_constant<T, Step>, integral_constant<T, NumSteps>> {
            GRIDTOOLS_STATIC_ASSERT(std::is_integral<T>::value, GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT(std::is_signed<T>::value, GT_INTERNAL_ERROR);

            template <class Fun>
            struct invocation_f {
                Fun m_fun;

                template <class Ptr, class Strides>
                void GT_FUNCTION operator()(Ptr &RESTRICT ptr, const Strides &RESTRICT strides) {
                    auto &&stride = get_stride<I>(strides);
                    for (T i = 0; i != NumSteps; ++i) {
                        m_fun(ptr, strides);
                        shift(ptr, stride, integral_constant<T, Step>{});
                    }
                    shift(ptr, stride, integral_constant<T, -Step * NumSteps>{});
                }
            };

          public:
            template <class U, class X>
            constexpr GT_FUNCTION loop(U &&, X &&) {}

            template <class Fun>
            constexpr GT_FUNCTION invocation_f<decay_t<Fun>> operator()(Fun &&fun) const {
                return {const_expr::forward<Fun>(fun)};
            }

            static constexpr size_t index = I;

            constexpr GT_FUNCTION integral_constant<T, Step> step() const { return {}; }
            constexpr GT_FUNCTION integral_constant<T, NumSteps> num_steps() const { return {}; }
        };

        template <size_t I, class T>
        class loop<I, integral_constant<T, 0>, integral_constant<T, 1>> {
            GRIDTOOLS_STATIC_ASSERT(std::is_integral<T>::value, GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT(std::is_signed<T>::value, GT_INTERNAL_ERROR);

          public:
            template <class U, class X>
            constexpr GT_FUNCTION loop(U &&, X &&) {}

            template <class Fun>
            constexpr GT_FUNCTION decay_t<Fun> operator()(Fun &&fun) const {
                return const_expr::forward<Fun>(fun);
            }

            static constexpr size_t index = I;

            constexpr GT_FUNCTION integral_constant<T, 0> step() const { return {}; }
            constexpr GT_FUNCTION integral_constant<T, 1> num_steps() const { return {}; }
        };

        template <size_t I, class T>
        class loop<I, integral_constant<T, 0>, integral_constant<T, 0>> {
            GRIDTOOLS_STATIC_ASSERT(std::is_integral<T>::value, GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT(std::is_signed<T>::value, GT_INTERNAL_ERROR);

          public:
            template <class U, class X>
            constexpr GT_FUNCTION loop(U &&, X &&) {}

            template <class Fun>
            constexpr GT_FUNCTION host_device::noop operator()(Fun &&fun) const {
                return {};
            }

            static constexpr size_t index = I;

            constexpr GT_FUNCTION integral_constant<T, 0> step() const { return {}; }
            constexpr GT_FUNCTION integral_constant<T, 0> num_steps() const { return {}; }
        };

        namespace loop_impl_ {

            template <class T, class = void>
            struct is_constant : std::false_type {};

            template <class T>
            struct is_constant<T,
                enable_if_t<std::is_base_of<std::integral_constant<typename T::value_type, T::value>, T>::value>>
                : std::true_type {};

            template <class T, class = void>
            struct value_type;

            template <class T>
            struct value_type<T, enable_if_t<std::is_integral<T>::value>> {
                using type = T;
            };

            template <class T>
            struct value_type<T, enable_if_t<is_constant<T>::value>> {
                using type = typename T::value_type;
            };

            template <class NumSteps, class T, bool = is_constant<NumSteps>::value>
            struct num_steps_type {
                using type = T;
            };

            template <class NumSteps, class T>
            struct num_steps_type<NumSteps, T, true> {
                using type = integral_constant<T, NumSteps::value>;
            };

            template <class Step, class NumSteps, class T, bool = is_constant<Step>::value>
            struct step_type {
                using type = T;
            };

            template <class Step, class T, bool IsStepConstant>
            struct step_type<Step, integral_constant<T, 0>, T, IsStepConstant> {
                using type = integral_constant<T, 0>;
            };

            template <class Step, class T, bool IsStepConstant>
            struct step_type<Step, integral_constant<T, 1>, T, IsStepConstant> {
                using type = integral_constant<T, 0>;
            };

            template <class Step, class NumSteps, class T>
            struct step_type<Step, NumSteps, T, true> {
                using type = integral_constant<T, Step::value>;
            };

            template <class T, bool = is_constant<T>::value>
            struct may_be_non_negative : std::true_type {};

            template <class T>
            struct may_be_non_negative<T, true> : bool_constant<(T::value >= 0)> {};

        } // namespace loop_impl_

        template <size_t I,
            class RawStep,
            class RawNumSteps,
            class T = make_signed_t<common_type_t<typename loop_impl_::value_type<RawStep>::type,
                typename loop_impl_::value_type<RawNumSteps>::type>>,
            class NumSteps = typename loop_impl_::num_steps_type<RawNumSteps, T>::type,
            class Step = typename loop_impl_::step_type<RawStep, NumSteps, T>::type,
            enable_if_t<loop_impl_::may_be_non_negative<RawNumSteps>::value, int> = 0>
        constexpr GT_FUNCTION loop<I, Step, NumSteps> make_loop(RawStep step, RawNumSteps num_steps) {
            return {step, num_steps};
        }

        template <class Loops>
        constexpr GT_FUNCTION auto nest_loops(Loops &&loops)
            GT_AUTO_RETURN(tuple_util::host_device::fold(host_device::compose{}, const_expr::forward<Loops>(loops)));

    } // namespace sid
} // namespace gridtools

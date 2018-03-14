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

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

#include "generic_metafunctions/meta.hpp"
#include "defs.hpp"
#include "functional.hpp"

namespace gridtools {

    namespace tuple_util {

        namespace traits {

            // std::tuple
            template < size_t I, class... Ts >
            constexpr typename std::tuple_element< I, std::tuple< Ts... > >::type &do_get(
                std::integral_constant< size_t, I >, std::tuple< Ts... > &obj) noexcept {
                return std::get< I >(obj);
            }
            template < size_t I, class... Ts >
            constexpr typename std::tuple_element< I, std::tuple< Ts... > >::type const &do_get(
                std::integral_constant< size_t, I >, std::tuple< Ts... > const &obj) noexcept {
                return std::get< I >(obj);
            }
            template < size_t I, class... Ts >
            constexpr typename std::tuple_element< I, std::tuple< Ts... > >::type &&do_get(
                std::integral_constant< size_t, I >, std::tuple< Ts... > &&obj) noexcept {
                return std::get< I >(std::move(obj));
            }

            // std::array
            template < size_t I, class T, size_t N >
            constexpr T &do_get(std::integral_constant< size_t, I >, std::array< T, N > &obj) noexcept {
                return std::get< I >(obj);
            }
            template < size_t I, class T, size_t N >
            constexpr T const &do_get(std::integral_constant< size_t, I >, std::array< T, N > const &obj) noexcept {
                return std::get< I >(obj);
            }
            template < size_t I, class T, size_t N >
            constexpr T &&do_get(std::integral_constant< size_t, I >, std::array< T, N > &&obj) noexcept {
                return std::get< I >(std::move(obj));
            }

            // std::pair
            template < size_t I, class T1, class T2 >
            constexpr typename std::tuple_element< I, std::pair< T1, T2 > >::type &do_get(
                std::integral_constant< size_t, I >, std::pair< T1, T2 > &obj) noexcept {
                return std::get< I >(obj);
            }
            template < size_t I, class T1, class T2 >
            constexpr const typename std::tuple_element< I, std::pair< T1, T2 > >::type &do_get(
                std::integral_constant< size_t, I >, const std::pair< T1, T2 > &obj) noexcept {
                return std::get< I >(obj);
            }
            template < size_t I, class T1, class T2 >
            constexpr typename std::tuple_element< I, std::pair< T1, T2 > >::type &&do_get(
                std::integral_constant< size_t, I >, std::pair< T1, T2 > &&obj) noexcept {
                return std::get< I >(std::move(obj));
            }

            template < size_t I, class T >
            constexpr auto get(T &&obj) noexcept GT_AUTO_RETURN(
                do_get(std::integral_constant< size_t, I >{}, std::forward< T >(obj)));
        }
        using traits::get;

        namespace _impl {

            template < class GeneratorList, class Res >
            struct generate_f;
            template < template < class... > class L, class... Generators, class Res >
            struct generate_f< L< Generators... >, Res > {
                template < class... Args >
                Res operator()(Args &&... args) const {
                    return {Generators{}(std::forward< Args >(args)...)...};
                }
            };

            enum class ref_kind { rvalue, lvalue, const_lvalue };

            template < class >
            struct get_ref_kind;

            template < class T >
            struct get_ref_kind< T && > : std::integral_constant< ref_kind, ref_kind::rvalue > {};

            template < class T >
            struct get_ref_kind< T & > : std::integral_constant< ref_kind, ref_kind::lvalue > {};

            template < class T >
            struct get_ref_kind< T const & > : std::integral_constant< ref_kind, ref_kind::const_lvalue > {};

            template < ref_kind Kind, class Dst >
            struct add_ref;

            template < class T >
            struct add_ref< ref_kind::rvalue, T > : std::add_rvalue_reference< T > {};

            template < class T >
            struct add_ref< ref_kind::lvalue, T > : std::add_lvalue_reference< T > {};

            template < class T >
            struct add_ref< ref_kind::const_lvalue, T >
                : std::add_lvalue_reference< typename std::add_const< T >::type > {};

            template < ref_kind Kind >
            struct get_accessor {
                template < class T >
                using apply = typename add_ref< Kind, T >::type;
            };

            template < class Tup >
            using get_accessors =
                meta::apply< meta::transform< get_accessor< get_ref_kind< Tup >::value >::template apply >,
                    typename std::decay< Tup >::type >;

            template < class Fun >
            struct get_fun_result {
                template < class... Ts >
                using apply = decltype(std::declval< Fun >()(std::declval< Ts >()...));
            };

            template < size_t I >
            struct transform_elem_f {
                template < class Fun, class... Tups >
                auto operator()(Fun &&fun, Tups &&... tups) const
                    GT_AUTO_RETURN(std::forward< Fun >(fun)(get< I >(std::forward< Tups >(tups)...)));
            };

            template < class Fun >
            struct transform_f {
                using get_results_t = meta::transform< get_fun_result< Fun >::template apply >;

                template < class I >
                using get_generator = transform_elem_f< I::value >;

                Fun m_fun;

                template < class Tup,
                    class... Tups,
                    class Res = meta::apply< get_results_t, get_accessors< Tup && >, get_accessors< Tups && >... > >
                Res operator()(Tup &&tup, Tups &&... tups) const {
                    constexpr auto length = meta::length< typename std::decay< Tup >::type >::value;
                    using generators = meta::apply< meta::transform< get_generator >, meta::make_indices< length > >;
                    return generate_f< generators, Res >{}(
                        m_fun, std::forward< Tup >(tup), std::forward< Tups >(tups)...);
                }
            };

            template < class >
            struct for_each_impl_f;

            template < template < class T, T... > class L, class Int, Int... Is >
            struct for_each_impl_f< L< Int, Is... > > {
                template < class Fun, class... Tups >
                void operator()(Fun &&fun, Tups &&... tups) const {
                    void((int[]){(std::forward< Fun >(fun)(get< Is >(std::forward< Tups >(tups)...)), 0)...});
                }
            };

            template < class Fun >
            struct for_each_f {
                Fun m_fun;

                template < class Lhs, class Rhs >
                using same_length = meta::bool_constant< meta::length< typename std::decay< Lhs >::type >::value ==
                                                         meta::length< typename std::decay< Rhs >::type >::value >;

                template < class Tup, class... Tups >
                typename std::enable_if< meta::conjunction< same_length< Tup, Tups >... >::value >::type operator()(
                    Tup &&tup, Tups &&... tups) const {
                    for_each_impl_f<
                        make_gt_index_sequence< meta::length< typename std::decay< Tup >::type >::value > >{}(
                        m_fun, std::forward< Tup >(tup), std::forward< Tups >(tups)...);
                }
            };

            struct flatten_f {
                template < size_t OuterI, size_t InnerI >
                struct generator_f {
                    template < class Tup >
                    auto operator()(Tup &&tup) const
                        GT_AUTO_RETURN(get< InnerI >(get< OuterI >(std::forward< Tup >(tup))));
                };

                template < class OuterI, class InnerI >
                using get_generator = generator_f< OuterI::value, InnerI::value >;

                template < class OuterI, class InnerTup >
                using get_inner_generators =
                    meta::apply< meta::transform< meta::bind< get_generator, OuterI, meta::_1 >::template apply >,
                        meta::make_indices_for< InnerTup > >;

                template < class Tup,
                    class Accessors = meta::apply< meta::transform< get_accessors >, get_accessors< Tup && > >,
                    class Res = meta::flatten< Accessors > >
                Res operator()(Tup &&tup) const {
                    using generators = meta::flatten< meta::apply< meta::transform< get_inner_generators >,
                        meta::make_indices_for< Accessors >,
                        Accessors > >;
                    return generate_f< generators, Res >{}(std::forward< Tup >(tup));
                }
            };

            template < size_t N >
            struct drop_front_f {
                template < size_t I >
                struct generator_f {
                    template < class Tup >
                    auto operator()(Tup &&tup) const GT_AUTO_RETURN(get< N + I >(std::forward< Tup >(tup)));
                };

                template < class I >
                using get_generator = generator_f< I::value >;

                template < class Tup,
                    class Accessors = get_accessors< Tup && >,
                    class Res = meta::drop_front< N, Accessors > >
                Res operator()(Tup &&tup) const {
                    using generators = meta::apply< meta::transform< get_generator >,
                        meta::make_indices< meta::length< Accessors >::value - N > >;
                    return generate_f< generators, Res >{}(std::forward< Tup >(tup));
                }
            };

            template < class, class >
            struct push_back_impl_f;

            template < template < class T, T... > class L, class Int, Int... Is, class Res >
            struct push_back_impl_f< L< Int, Is... >, Res > {
                template < class Tup, class... Args >
                Res operator()(Tup &&tup, Args &&... args) const {
                    return {get< Is >(std::forward< Tup >(tup))..., std::forward< Args >(args)...};
                }
            };

            struct push_back_f {
                template < class Tup,
                    class... Args,
                    class Accessors = get_accessors< Tup && >,
                    class Res = meta::push_back< Accessors, Args &&... > >
                Res operator()(Tup &&tup, Args &&... args) const {
                    return push_back_impl_f< make_gt_index_sequence< meta::length< Accessors >::value >, Res >{}(
                        std::forward< Tup >(tup), std::forward< Args >(args)...);
                }
            };

            template < class Fun >
            struct fold_f {

                template < class S, class T >
                using meta_fun = meta::apply< get_fun_result< Fun >, S, T >;

                Fun m_fun;

                template < class State, template < class... > class L >
                State operator()(State &&state, L<>) const {
                    return state;
                }

                template < class State,
                    class Tup,
                    class Accessors = get_accessors< Tup && >,
                    class Res = meta::apply< meta::lfold< meta_fun >, State &&, Accessors > >
                Res operator()(State &&state, Tup &&tup) const {
                    auto &&new_state = m_fun(std::forward< State >(state), get< 0 >(std::forward< Tup >(tup)));
                    auto &&rest = drop_front_f< 1 >{}(std::forward< Tup >(tup));
                    return this->operator()(std::move(new_state), std::move(rest));
                }

                template < class Tup >
                auto operator()(Tup &&tup) const GT_AUTO_RETURN(this->operator()(
                    get< 0 >(std::forward< Tup >(tup)), drop_front_f< 1 >{}(std::forward< Tup >(tup))));
            };
        }

        template < class Fun >
        constexpr _impl::transform_f< Fun > transform(Fun fun) {
            return {std::move(fun)};
        }

        template < class Fun, class Tup >
        auto transform(Fun &&fun, Tup &&tup)
            GT_AUTO_RETURN(_impl::transform_f< Fun >{std::forward< Fun >(fun)}(std::forward< Tup >(tup)));

        template < class Fun >
        constexpr _impl::for_each_f< Fun > for_each(Fun fun) {
            return {std::move(fun)};
        }

        template < class Fun, class Tup >
        void for_each(Fun &&fun, Tup &&tup) {
            _impl::for_each_f< Fun >{std::forward< Fun >(fun)}(std::forward< Tup >(tup));
        }

        inline constexpr _impl::flatten_f flatten() { return {}; }

        template < class Tup >
        auto flatten(Tup &&tup) GT_AUTO_RETURN(flatten()(std::forward< Tup >(tup)));

        template < class Generators, class Res, class... Args >
        Res generate(Args &&... args) {
            return _impl::generate_f< Generators, Res >{}(std::forward< Args >(args)...);
        }

        template < size_t N >
        constexpr _impl::drop_front_f< N > drop_front() {
            return {};
        }

        template < size_t N, class Tup >
        auto drop_front(Tup &&tup) GT_AUTO_RETURN(drop_front< N >()(std::forward< Tup >(tup)));

        inline constexpr _impl::push_back_f push_back() { return {}; }

        template < class Tup, class... Args >
        auto push_back(Tup &&tup, Args &&... args)
            GT_AUTO_RETURN(push_back()(std::forward< Tup >(tup), std::forward< Args >(args)...));

        template < class Fun >
        constexpr _impl::fold_f< Fun > fold(Fun fun) {
            return {std::move(fun)};
        }

        template < class Fun, class Arg, class... Args >
        auto fold(Fun &&fun, Arg &&arg, Args &&... args) GT_AUTO_RETURN(
            _impl::fold_f< Fun >{std::forward< Fun >(fun)}(std::forward< Arg >(arg), std::forward< Args >(args)...));

        inline constexpr _impl::transform_f< clone > deep_copy() { return {}; }

        template < class Tup >
        auto deep_copy(Tup &&tup) GT_AUTO_RETURN(deep_copy()(std::forward< Tup >(tup)));
    }
}

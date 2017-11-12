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

#include <tuple>
#include <functional>
#include "../common/halo_descriptor.hpp"
#include "../boundary-conditions/boundary.hpp"

namespace gridtools {
    namespace _impl {

        struct Plc {};
        struct NotPlc {};

        template < int V >
        struct PlcOrNot {
            using type = Plc;
        };

        template <>
        struct PlcOrNot< 0 > {
            using type = NotPlc;
        };

        template < uint_t I, typename ROTuple, typename AllTuple >
        auto select_element(ROTuple const &ro_tuple, AllTuple const &all, Plc) -> decltype(
            std::get< std::is_placeholder< typename std::tuple_element< I, AllTuple >::type >::value - 1 >(ro_tuple)) {
            return std::get< std::is_placeholder< typename std::tuple_element< I, AllTuple >::type >::value - 1 >(
                ro_tuple);
        }

        template < uint_t I, typename ROTuple, typename AllTuple >
        auto select_element(ROTuple const &ro_tuple, AllTuple const &all, NotPlc) -> decltype(std::get< I >(all)) {
            return std::get< I >(all);
        }

        template < typename ROTuple, typename AllTuple, uint_t... IDs >
        auto substitute_placeholders(
            ROTuple const &ro_tuple, AllTuple const &all, gt_integer_sequence< uint_t, IDs... >)
            -> decltype(std::make_tuple(select_element< IDs >(
                ro_tuple,
                all,
                typename PlcOrNot<
                    std::is_placeholder< typename std::tuple_element< IDs, AllTuple >::type >::value >::type{})...)) {
            return std::make_tuple(select_element< IDs >(
                ro_tuple,
                all,
                typename PlcOrNot<
                    std::is_placeholder< typename std::tuple_element< IDs, AllTuple >::type >::value >::type{})...);
        }

        std::tuple<> rest_tuple(std::tuple<>, gt_integer_sequence< uint_t >) { return {}; }

        template < typename... Elems, uint_t... IDs >
        auto rest_tuple(std::tuple< Elems... > const &x, gt_integer_sequence< uint_t, IDs... >)
            -> decltype(std::make_tuple(std::get< IDs + 1u >(x)...)) {
            return std::make_tuple(std::get< IDs + 1u >(x)...);
        }

        std::tuple<> do_stuff(std::tuple<> const &) { return {}; }

        template < typename First >
        std::tuple< First > do_stuff(std::tuple< First > const &x,
            typename std::enable_if< std::is_placeholder< First >::value == 0, void * >::type = nullptr) {
            return {x};
        }

        template < typename First >
        std::tuple<> do_stuff(std::tuple< First > const &x,
            typename std::enable_if< (std::is_placeholder< First >::value > 0), void * >::type = nullptr) {
            return {};
        }

        template < typename First, typename... Elems >
        auto do_stuff(std::tuple< First, Elems... > const &x,
            typename std::enable_if< std::is_placeholder< First >::value == 0, void * >::type = nullptr)
            -> decltype(std::tuple_cat(std::make_tuple(std::get< 0 >(x)),
                do_stuff(rest_tuple(x, typename make_gt_integer_sequence< uint_t, sizeof...(Elems) >::type{})))) {
            return std::tuple_cat(std::make_tuple(std::get< 0 >(x)),
                do_stuff(rest_tuple(x, typename make_gt_integer_sequence< uint_t, sizeof...(Elems) >::type{})));
        }

        template < typename First, typename... Elems >
        auto do_stuff(std::tuple< First, Elems... > const &x,
            typename std::enable_if< (std::is_placeholder< First >::value > 0), void * >::type = nullptr)
            -> decltype(
                do_stuff(rest_tuple(x, typename make_gt_integer_sequence< uint_t, sizeof...(Elems) >::type{}))) {
            return do_stuff(rest_tuple(x, typename make_gt_integer_sequence< uint_t, sizeof...(Elems) >::type{}));
        }

        template < typename TupleWithPlcs >
        auto remove_placeholders(TupleWithPlcs const &with_plcs) -> decltype(do_stuff(with_plcs)) {
            return do_stuff(with_plcs);
        }
    } // namespace _impl

    template < typename BCApply, typename DataStores, typename ExcStores >
    struct binded_bc {
        using boundary_class = BCApply;
        boundary_class m_bcapply;
        using stores_type = DataStores;
        stores_type m_stores;
        using exc_stores_type = ExcStores;
        exc_stores_type m_exc_stores;

        binded_bc(BCApply bca, stores_type stores_list, exc_stores_type exc_stores_list)
            : m_bcapply{bca}, m_stores{stores_list}, m_exc_stores{exc_stores_list} {}

        stores_type stores() { return m_stores; }
        stores_type const stores() const { return m_stores; }

        exc_stores_type exc_stores() { return m_exc_stores; }
        stores_type const exc_stores() const { return m_exc_stores; }

        boundary_class boundary_to_apply() const { return m_bcapply; }

        template < typename... ReadOnly >
        auto associate(ReadOnly... ro_stores) const -> binded_bc< BCApply,
            decltype(_impl::substitute_placeholders(std::make_tuple(ro_stores...),
                m_stores,
                typename make_gt_integer_sequence< uint_t, std::tuple_size< decltype(m_stores) >::value >::type{})),
            decltype(_impl::remove_placeholders(m_stores)) > {
            auto ro_store_tuple = std::make_tuple(ro_stores...);
            // we need to substitute the placeholders with the
            auto full_list = _impl::substitute_placeholders(ro_store_tuple,
                m_stores,
                typename make_gt_integer_sequence< uint_t, std::tuple_size< decltype(m_stores) >::value >::type{});
            auto without_plcs = _impl::remove_placeholders(m_stores);

            return binded_bc< BCApply, decltype(full_list), decltype(without_plcs) >(
                m_bcapply, full_list, without_plcs);
        }
    };

    template < typename BCApply, typename... DataStores >
    binded_bc< BCApply, std::tuple< DataStores... >, std::tuple< DataStores... > > bind_bc(
        BCApply bc_apply, DataStores... stores) {
        return {bc_apply, std::make_tuple(stores...), std::make_tuple(stores...)};
    }

    template < typename T >
    struct is_binded_bc {
        static constexpr bool value = false;
    };

    template < typename... T >
    struct is_binded_bc< binded_bc< T... > > {
        static constexpr bool value = true;
    };

} // namespace gridtools

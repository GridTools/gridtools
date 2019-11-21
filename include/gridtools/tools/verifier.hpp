/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <iostream>
#include <memory>
#include <type_traits>

#include "../common/array.hpp"
#include "../common/array_addons.hpp"
#include "../common/gt_math.hpp"
#include "../common/hypercube_iterator.hpp"
#include "../common/tuple_util.hpp"
#include "../storage/data_store.hpp"

namespace gridtools {
    namespace impl_ {
        template <class T>
        struct default_precision_impl {
            static constexpr double value = 0;
        };

        template <>
        struct default_precision_impl<float> {
            static constexpr double value = 1e-6;
        };

        template <>
        struct default_precision_impl<double> {
            static constexpr double value = 1e-14;
        };
    } // namespace impl_

    template <class T>
    GT_FUNCTION double default_precision() {
        return impl_::default_precision_impl<T>::value;
    }

    template <class T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
    GT_FUNCTION bool expect_with_threshold(T expected, T actual, double precision = default_precision<T>()) {
        auto abs_error = math::fabs(expected - actual);
        auto abs_max = math::max(math::fabs(expected), math::fabs(actual));
        return abs_error < precision || abs_error < abs_max * precision;
    }

    template <class T, std::enable_if_t<!std::is_floating_point<T>::value, int> = 0>
    GT_FUNCTION bool expect_with_threshold(T const &expected, T const &actual, double = 0) {
        return actual == expected;
    }

    namespace verify_impl_ {
        template <class F, class Indices, size_t... Is>
        auto apply_impl(F const &fun, Indices const &indices, std::index_sequence<Is...>)
            -> decltype(fun(tuple_util::get<Is>(indices)...)) {
            return fun(tuple_util::get<Is>(indices)...);
        }
        template <class F, class Indices>
        auto apply(F const &fun, Indices const &indices)
            -> decltype(apply_impl(fun, indices, std::make_index_sequence<tuple_util::size<Indices>::value>())) {
            return apply_impl(fun, indices, std::make_index_sequence<tuple_util::size<Indices>::value>());
        }

        template <class F, size_t N, class = void>
        struct is_view_compatible : std::false_type {};

        template <class F, size_t N>
        struct is_view_compatible<F, N, void_t<decltype(apply(std::declval<F const &>(), array<size_t, N>{}))>>
            : std::true_type {};

        template <class Expected, class DataStore, class Halos>
        std::enable_if_t<storage::is_data_store<DataStore>::value &&
                             is_view_compatible<Expected, DataStore::ndims>::value,
            bool>
        verify_data_store(
            Expected const &expected, std::shared_ptr<DataStore> const &actual, Halos const &halos, double precision) {
            array<array<size_t, 2>, DataStore::ndims> bounds;
            auto &&lengths = actual->lengths();
            for (size_t i = 0; i < bounds.size(); ++i)
                bounds[i] = {halos[i][0], lengths[i] - halos[i][1]};
            auto view = actual->const_host_view();
            static constexpr size_t err_lim = 20;
            size_t err_count = 0;
            for (auto &&pos : make_hypercube_view(bounds)) {
                auto a = apply(view, pos);
                decltype(a) e = apply(expected, pos);
                if (expect_with_threshold(e, a, precision))
                    continue;
                if (err_count < err_lim)
                    std::cout << "Error in position " << pos << " ; expected : " << e << " ; actual : " << a << "\n";
                err_count++;
            }
            if (err_count > err_lim)
                std::cout << "Displayed the first " << err_lim << " errors, " << err_count - err_lim << " skipped!"
                          << std::endl;
            return err_count == 0;
        }

        template <class DataStore, class Halos>
        std::enable_if_t<storage::is_data_store_ptr<DataStore>::value, bool> verify_data_store(
            DataStore const &expected, DataStore const &actual, Halos const &halos, double precision) {
            return verify_data_store(expected->const_host_view(), actual, halos, precision);
        }

        template <class T, class DataStore, class Halos>
        std::enable_if_t<storage::is_data_store<DataStore>::value &&
                             std::is_convertible<T, typename DataStore::data_t>::value,
            bool>
        verify_data_store(
            T const &expected, std::shared_ptr<DataStore> const &actual, Halos const &halos, double precision) {
            return verify_data_store([=](auto &&...) { return expected; }, actual, halos, precision);
        }

        template <class Expected, class DataStore, class Halos>
        bool verify_data_store(Expected const &expected, DataStore const &actual, Halos const &halos) {
            return verify_data_store(expected, actual, halos, default_precision<DataStore::element_type::data_t>());
        }

        template <class Expected, class DataStore>
        bool verify_data_store(Expected const &expected, DataStore const &actual, double const &precision) {
            return verify(expected, actual, array<array<size_t, 2>, DataStore::element_type::ndims>{}, precision);
        }

        template <class Expected, class DataStore>
        bool verify_data_store(Expected const &expected, DataStore const &actual) {
            return verify(expected,
                actual,
                array<array<size_t, 2>, DataStore::element_type::ndims>{},
                default_precision<DataStore::data_t>());
        }
    } // namespace verify_impl_
    using verify_impl_::verify_data_store;
} // namespace gridtools

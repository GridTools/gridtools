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

#include <gridtools/common/array.hpp>
#include <gridtools/common/array_addons.hpp>
#include <gridtools/common/gt_math.hpp>
#include <gridtools/common/hypercube_iterator.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/storage/data_store.hpp>

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
        auto apply(F const &fun, Indices const &indices) -> decltype(
            verify_impl_::apply_impl(fun, indices, std::make_index_sequence<tuple_util::size<Indices>::value>())) {
            return apply_impl(fun, indices, std::make_index_sequence<tuple_util::size<Indices>::value>());
        }

        template <class F, size_t N, class = void>
        struct is_view_compatible : std::false_type {};

        template <class F, size_t N>
        struct is_view_compatible<F,
            N,
            void_t<decltype(verify_impl_::apply(std::declval<F const &>(), array<size_t, N>{}))>> : std::true_type {};

        struct float_equal_to {
            template <class T>
            bool operator()(T lhs, T rhs) const {
                return expect_with_threshold(lhs, rhs);
            }
        };

        template <class DataStore>
        using default_equal_to =
            meta::if_<std::is_floating_point<typename DataStore::data_t>, float_equal_to, std::equal_to<>>;

        template <class Expected, class DataStore, class Halos, class EqualTo = default_equal_to<DataStore>>
        std::enable_if_t<storage::is_data_store<DataStore>::value &&
                             is_view_compatible<Expected, DataStore::ndims>::value,
            bool>
        verify_data_store(Expected const &expected,
            std::shared_ptr<DataStore> const &actual,
            Halos const &halos,
            EqualTo equal_to = {}) {
            array<array<size_t, 2>, DataStore::ndims> bounds;
            auto &&lengths = actual->lengths();
            for (size_t i = 0; i < bounds.size(); ++i)
                bounds[i] = {halos[i][0], lengths[i] - halos[i][1]};
            auto view = actual->const_host_view();
            static constexpr size_t err_lim = 20;
            size_t err_count = 0;
            for (auto &&pos : make_hypercube_view(bounds)) {
                auto a = verify_impl_::apply(view, pos);
                decltype(a) e = verify_impl_::apply(expected, pos);
                if (equal_to(e, a))
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

        template <class DataStore, class Halos, class EqualTo = default_equal_to<DataStore>>
        std::enable_if_t<storage::is_data_store_ptr<DataStore>::value, bool> verify_data_store(
            DataStore const &expected, DataStore const &actual, Halos const &halos, EqualTo equal_to = {}) {
            return verify_data_store(expected->const_host_view(), actual, halos, equal_to);
        }

        template <class T, class DataStore, class Halos, class EqualTo = default_equal_to<DataStore>>
        std::enable_if_t<storage::is_data_store<DataStore>::value &&
                             std::is_convertible<T, typename DataStore::data_t>::value,
            bool>
        verify_data_store(
            T const &expected, std::shared_ptr<DataStore> const &actual, Halos const &halos, EqualTo equal_to = {}) {
            return verify_data_store([=](auto &&...) { return expected; }, actual, halos, equal_to);
        }
    } // namespace verify_impl_
    using verify_impl_::default_equal_to;
    using verify_impl_::verify_data_store;
} // namespace gridtools

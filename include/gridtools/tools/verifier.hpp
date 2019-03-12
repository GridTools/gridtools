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
#include <type_traits>

#include "../common/array.hpp"
#include "../common/array_addons.hpp"
#include "../common/gt_math.hpp"
#include "../common/hypercube_iterator.hpp"
#include "../common/tuple_util.hpp"
#include "../meta/type_traits.hpp"
#include "../stencil_composition/grid_traits_fwd.hpp"
#include "../storage/common/storage_info_rt.hpp"
#include "../storage/storage_facility.hpp"

namespace gridtools {

    namespace impl_ {
        template <class T>
        class default_precision_impl;

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

    template <typename T, enable_if_t<std::is_floating_point<T>::value, int> = 0>
    GT_FUNCTION bool expect_with_threshold(T expected, T actual, double precision = default_precision<T>()) {
        auto abs_error = math::fabs(expected - actual);
        auto abs_max = math::max(math::fabs(expected), math::fabs(actual));
        return abs_error < precision || abs_error < abs_max * precision;
    }

    template <typename T, typename Dummy = int, enable_if_t<!std::is_floating_point<T>::value, int> = 0>
    GT_FUNCTION bool expect_with_threshold(T const &expected, T const &actual, Dummy = 0) {
        return actual == expected;
    }

    class verifier {
        double m_precision;
        size_t m_max_error;

      public:
        verifier(double precision, size_t max_error = 20) : m_precision(precision), m_max_error(max_error) {}

        template <typename Grid, typename StorageType>
        bool verify(Grid const &grid_ /*TODO: unused*/,
            StorageType const &expected_field,
            StorageType const &actual_field,
            array<array<uint_t, 2>, StorageType::storage_info_t::layout_t::masked_length> halos = {}) {
            // TODO This is following the original implementation. Shouldn't we deduce the range from the grid (as we
            // already pass it)?
            storage_info_rt meta_rt = make_storage_info_rt(*(expected_field.get_storage_info_ptr()));
            array<array<size_t, 2>, StorageType::storage_info_t::layout_t::masked_length> bounds;
            for (size_t i = 0; i < bounds.size(); ++i) {
                bounds[i] = {halos[i][0], meta_rt.total_lengths()[i] - halos[i][1]};
            }
            auto cube_view = make_hypercube_view(bounds);

            expected_field.sync();
            auto expected_view = make_host_view<access_mode::read_only>(expected_field);
            actual_field.sync();
            auto actual_view = make_host_view<access_mode::read_only>(actual_field);

            size_t error_count = 0;
            for (auto &&pos : cube_view) {
                auto expected = expected_view(tuple_util::convert_to<array, int>(pos));
                auto actual = actual_view(tuple_util::convert_to<array, int>(pos));
                if (!expect_with_threshold(expected, actual, m_precision)) {
                    if (error_count < m_max_error)
                        std::cout << "Error in position " << pos << " ; expected : " << expected
                                  << " ; actual : " << actual << "\n";
                    error_count++;
                }
            }
            if (error_count > m_max_error)
                std::cout << "Displayed the first " << m_max_error << " errors, " << error_count - m_max_error
                          << " skipped!" << std::endl;
            return error_count == 0;
        }
    };

} // namespace gridtools

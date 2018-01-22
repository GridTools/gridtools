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

#include "common/multi_iterator.hpp"
#include "common/array.hpp"
#include "common/make_array.hpp"
#include "common/gt_math.hpp"
#include "stencil-composition/grid_traits_fwd.hpp"
#include "storage/common/storage_info_rt.hpp"

namespace gridtools {

    template < typename value_type >
    GT_FUNCTION bool compare_below_threshold(value_type expected, value_type actual, double precision) {
        value_type absmax = math::max(math::fabs(expected), math::fabs(actual));
        value_type absolute_error = math::fabs(expected - actual);
        value_type relative_error = absolute_error / absmax;
        if (relative_error <= precision || absolute_error < precision) {
            return true;
        }
        return false;
    }

    class verifier {
      private:
        // Can be replaced by a lambda with auto... in c++14
        template < typename ExpectedView, typename ActualView >
        struct verify_functor {
            const ExpectedView &expected_view;
            const ActualView &actual_view;
            double precision_;

            verify_functor(const ExpectedView &expected_view, const ActualView &actual_view, double precision)
                : expected_view(expected_view), actual_view(actual_view), precision_(precision) {}

            template < typename... Pos >
            bool operator()(Pos... pos) {
                auto expected = expected_view(pos...);
                auto actual = actual_view(pos...);
                if (!compare_below_threshold(expected, actual, precision_)) {
                    std::cout << "Error in position " << make_array(pos...) << " ; expected : " << expected
                              << " ; actual : " << actual << "  " << std::fabs((expected - actual) / (expected))
                              << std::endl;
                    return false;
                } else
                    return true;
            }
        };

        double m_precision;

      public:
        verifier(const double precision) : m_precision(precision) {}
        ~verifier() {}

        template < typename Grid, typename StorageType >
        bool verify(Grid const &grid_ /*TODO: unused*/,
            StorageType const &expected_field,
            StorageType const &actual_field,
            const array< array< uint_t, 2 >, StorageType::storage_info_t::layout_t::masked_length > halos) {
            if (StorageType::num_of_storages > 1)
                throw std::runtime_error("Verifier not supported for data fields with more than 1 components");

            auto expected_view = make_host_view(expected_field);
            auto actual_view = make_host_view(actual_field);

            // TODO This is following the original implementation. Shouldn't we deduce the range from the grid (as we
            // already pass it)?
            storage_info_rt meta_rt = make_storage_info_rt(*(expected_field.get_storage_info_ptr()));
            array< pair< uint_t, uint_t >, StorageType::storage_info_t::layout_t::masked_length > bounds;
            for (size_t i = 0; i < bounds.size(); ++i) {
                bounds[i] = {halos[i][0], meta_rt.unaligned_dims()[i] - halos[i][1]};
            }

            auto reduction_op = [](bool a, bool b) { return a && b; };

            return make_multi_iterator(bounds).reduce(verify_functor< decltype(expected_view), decltype(actual_view) >(
                                                          expected_view, actual_view, m_precision),
                reduction_op,
                true);
        }
    };

} // namespace gridtools

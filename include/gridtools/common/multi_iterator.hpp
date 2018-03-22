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

#include "defs.hpp"
#include "host_device.hpp"
#include "array.hpp"
#include "generic_metafunctions/meta.hpp"
#include "generic_metafunctions/is_all_integrals.hpp"
#include "array_transpose.hpp"

namespace gridtools {
    class range {
      private:
        array< size_t, 2 > m_range;

      public:
        range() = default;
        GT_FUNCTION range(size_t b, size_t e) : m_range{b, e} {}

        GT_FUNCTION
        constexpr size_t const &operator[](size_t i) const { return m_range[i]; }

        constexpr size_t const &begin() const { return m_range[0]; }
        constexpr size_t const &end() const { return m_range[1]; }
    };

    template < typename T >
    class tuple_size;
    template <>
    class tuple_size< range > : public gridtools::static_size_t< 2 > {};

    template < size_t I >
    GT_FUNCTION constexpr const size_t &get(const range &arr) noexcept {
        GRIDTOOLS_STATIC_ASSERT(I < 2, "index is out of bounds");
        return arr[I];
    }

    template < size_t D >
    using hypercube = array< range, D >;

    // TODO should end be included?
    template < size_t D >
    class hypercube_view {
      private:
        struct grid_iterator {
            array< size_t, D > pos_;
            array< size_t, D > begin_;
            array< size_t, D > end_;

            GT_FUNCTION grid_iterator(
                const array< size_t, D > &pos, const array< size_t, D > &begin, const array< size_t, D > &end)
                : pos_(pos), begin_(begin), end_(end) {}

            GT_FUNCTION grid_iterator &operator++() {
                for (size_t i = 0; i < D; ++i) {
                    size_t index = D - i - 1;
                    if (pos_[index] + 1 < end_[index]) {
                        pos_[index]++;
                        return *this;
                    } else {
                        pos_[index] = begin_[index];
                    }
                }
                // we reached the end
                for (size_t i = 0; i < D; ++i)
                    pos_[i] = end_[i];
                return *this;
            }

            GT_FUNCTION grid_iterator operator++(int) {
                grid_iterator tmp(*this);
                operator++();
                return tmp;
            }

            GT_FUNCTION array< size_t, D > const &operator*() const { return pos_; }

            GT_FUNCTION bool operator==(const grid_iterator &other) const { return pos_ == other.pos_; }

            GT_FUNCTION bool operator!=(const grid_iterator &other) const { return !operator==(other); }
        };

      public:
        template < typename PairType >
        GT_FUNCTION hypercube_view(const array< PairType, D > &cube)
            : iteration_space_{transpose(cube)} {}

        /**
         * Construct hypercube starting from 0.
         */
        //        GT_FUNCTION hypercube_view(const array< size_t, D > &sizes) : iteration_space_{array< size_t, D >{},
        //        sizes} {}

        GT_FUNCTION grid_iterator begin() const {
            return grid_iterator{iteration_space_[begin_], iteration_space_[begin_], iteration_space_[end_]};
        }
        GT_FUNCTION grid_iterator end() const {
            return grid_iterator{iteration_space_[end_], iteration_space_[begin_], iteration_space_[end_]};
        }

      private:
        array< array< size_t, D >, 2 > iteration_space_;
        const size_t begin_ = 0;
        const size_t end_ = 1;
    };

    namespace impl_ {
        template < typename... RangeTypes >
        using is_all_range =
            meta::conjunction< std::integral_constant< bool, tuple_size< RangeTypes >::value == 2 >... >;
    }

    template < typename... Range, typename std::enable_if< impl_::is_all_range< Range... >::value, int >::type = 0 >
    GT_FUNCTION auto make_hypercube_view(Range... r) GT_AUTO_RETURN(hypercube_view< sizeof...(Range) >(
        array< typename std::common_type< Range... >::type, sizeof...(Range) >{r...}));

    /**
     * Overload where range... = {0, size}...
     */
    //    template < typename... IntegerTypes,
    //        typename std::enable_if< is_all_integral< IntegerTypes... >::value, int >::type = 0 >
    //    GT_FUNCTION auto make_hypercube_view(IntegerTypes... size) GT_AUTO_RETURN(
    //        hypercube_view< sizeof...(IntegerTypes) >(hypercube< sizeof...(IntegerTypes) >{range(0, size)...}));

    template < size_t D >
    GT_FUNCTION auto make_hypercube_view(const hypercube< D > &cube) GT_AUTO_RETURN(hypercube_view< D >(cube));

    //    template < size_t D >
    //    GT_FUNCTION auto make_hypercube_view(const array< size_t, D > &sizes) GT_AUTO_RETURN(hypercube_view< D
    //    >(sizes));
}

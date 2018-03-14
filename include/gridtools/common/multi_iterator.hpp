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

namespace gridtools {
    class range : public array< size_t, 2 > {
      public:
        range() = default;
        GT_FUNCTION range(size_t b, size_t e) : array{{b, e}} {}
        GT_FUNCTION size_t begin() const { return this->operator[](0); }
        GT_FUNCTION size_t end() const { return this->operator[](1); }
    };

    template < size_t D >
    using hypercube = array< range, D >;

    namespace impl_ {
        /**
         * @brief returns the I-th entry of each element (array or tuple) of the array as an array
         */
        template < size_t I, size_t D, typename Pair >
        GT_FUNCTION array< size_t, D > transpose(const array< Pair, D > &container_of_pairs) {
            array< size_t, D > tmp;
            for (size_t i = 0; i < D; ++i)
                tmp[i] = get< I >(container_of_pairs[i]);
            return tmp;
        }

        template < size_t D >
        GT_FUNCTION auto begin_of_hypercube(const hypercube< D > &cube) GT_AUTO_RETURN(transpose< 0 >(cube));
        template < size_t D >
        GT_FUNCTION auto end_of_hypercube(const hypercube< D > &cube) GT_AUTO_RETURN(transpose< 1 >(cube));
    }

    // TODO should end be included?
    template < size_t D >
    class hypercube_view {
      public:
        GT_FUNCTION hypercube_view(const hypercube< D > &cube)
            : begin_{impl_::begin_of_hypercube(cube)}, end_{impl_::end_of_hypercube(cube)} {}

        struct grid_iterator {
            array< size_t, D > pos_;
            const array< size_t, D > begin_;
            const array< size_t, D > end_;

            GT_FUNCTION grid_iterator(
                const array< size_t, D > &pos, const array< size_t, D > &begin, const array< size_t, D > &end)
                : pos_{pos}, begin_{begin}, end_{end} {}

            GT_FUNCTION operator array< size_t, D >() const { return pos_; }

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

            GT_FUNCTION array< size_t, D > &operator*() { return pos_; }

            GT_FUNCTION bool operator==(const grid_iterator &other) const { return pos_ == other.pos_; }

            GT_FUNCTION bool operator!=(const grid_iterator &other) const { return !operator==(other); }
        };

        GT_FUNCTION grid_iterator begin() const { return grid_iterator{begin_, begin_, end_}; }
        GT_FUNCTION grid_iterator end() const { return grid_iterator{end_, begin_, end_}; }

        GT_FUNCTION bool operator==(const hypercube_view &other) const {
            return begin_ == other.begin_ && end_ == other.end_;
        }

      private:
        array< size_t, D > begin_;
        array< size_t, D > end_;
    };

    template < typename... Range > // TODO assert all types supporting range concept
    GT_FUNCTION auto make_hypercube_view(Range... r)
        GT_AUTO_RETURN(hypercube_view< sizeof...(Range) >(hypercube< sizeof...(Range) >{r...}));

    template < size_t D >
    GT_FUNCTION auto make_hypercube_view(const hypercube< D > &cube) GT_AUTO_RETURN(hypercube_view< D >(cube));
}

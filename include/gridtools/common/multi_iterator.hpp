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

#include "array.hpp"
#include "make_array.hpp"
#include "array_addons.hpp"
#include "generic_metafunctions/gt_integer_sequence.hpp"
#include "pair.hpp"
#include "defs.hpp"
#include "host_device.hpp"
#include "generic_metafunctions/meta.hpp"
#include "generic_metafunctions/is_all_integrals.hpp"

namespace gridtools {
    template < size_t D >
    using hypercube = array< pair< size_t, size_t >, D >;

    namespace impl_ {

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
            GT_FUNCTION hypercube_view(const array< array< size_t, D >, 2 > &iteration_space)
                : iteration_space_(iteration_space) {}

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
    }

    /**
     * @brief constructs a view on a hypercube from an array of ranges (e.g. pairs); the end of the range is exclusive
     */
    template < typename Container >
    GT_FUNCTION auto make_hypercube_view(Container &&cube)
        GT_AUTO_RETURN(impl_::hypercube_view< tuple_size< typename std::decay< Container >::type >::value >(
            transpose(std::forward< Container >(cube))));

    /**
     * @brief constructs a view on a hypercube from an array of integers (ranges start from 0); the end of the range is
     * exclusive
     */
    template < typename Container >
    GT_FUNCTION auto make_hypercube_view_from_zero(Container &&sizes)
        GT_AUTO_RETURN(impl_::hypercube_view< tuple_size< typename std::decay< Container >::type >::value >(
            array< array< size_t, tuple_size< Container >::value >, 2 >{
                array< size_t, tuple_size< Container >::value >{},
                convert_to< size_t >(std::forward< Container >(sizes))}));
}

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

#include "generic_metafunctions/gt_integer_sequence.hpp"
#include "generic_metafunctions/is_all_integrals.hpp"
#include "pair.hpp"
#include "array.hpp"
#include "array_addons.hpp"
#include "defs.hpp"
#include "host_device.hpp"

namespace gridtools {

    template < typename T >
    class range {
      private:
        T begin_;
        T end_;

      public:
        range(T b, T e) : begin_{b}, end_{e} {}
        GT_FUNCTION T begin() const { return begin_; }
        GT_FUNCTION T end() const { return end_; }
    };

    //    TODO begin() -> std::get<0>

    //    using range

    template < typename T >
    range< T > make_range(T b, T e) {
        return range< T >{b, e};
    }

    template < typename T, size_t D >
    class hypercube : public array< range< T >, D > {
      public:
        template < typename... T2, typename = all_integral< T2... > >
        GT_FUNCTION hypercube(const range< T2 > &... r)
            : array< range< T >, D >{{r...}} {}

        array< T, D > begin() const {
            array< T, D > tmp;
            for (T i = 0; i < D; ++i)
                tmp[i] = this->operator[](i).begin();
            return tmp;
        }
        array< T, D > end() const {
            array< T, D > tmp;
            for (T i = 0; i < D; ++i)
                tmp[i] = this->operator[](i).end();
            return tmp;
        }
    };

    namespace impl_ {
        /**
         * @brief returns the I-th entry of each element of a container as an array
         */
        template < typename Container, size_t D, size_t I >
        array< size_t, tuple_size< Container >::value > transpose(const Container &container_of_pairs) {
            array< size_t, tuple_size< Container >::value > tmp;
            for (size_t i = 0; i < tuple_size< Container >::value; ++i)
                tmp[i] = get< I >(container_of_pairs);
            return tmp;
        }
    }

    // TODO should end be included?
    template < typename T, size_t D >
    class hypercube_view {
      public:
        hypercube_view(const hypercube< T, D > &range) : begin_{range.begin()}, end_{range.end()} {} // TODO transpose

        struct grid_iterator {
            array< T, D > pos_;
            const array< T, D > begin_;
            const array< T, D > end_;

            grid_iterator(const array< T, D > &pos, const array< T, D > &begin, const array< T, D > &end)
                : pos_{pos}, begin_{begin}, end_{end} {}

            operator array< T, D >() const { return pos_; }

            grid_iterator &operator++() {
                for (T i = 0; i < D; ++i) {
                    T index = D - i - 1;
                    if (pos_[index] + 1 < end_[index]) {
                        pos_[index]++;
                        return *this;
                    } else {
                        pos_[index] = begin_[index];
                    }
                }
                // we reached the end
                for (T i = 0; i < D; ++i)
                    pos_[i] = end_[i];
                return *this;
            }

            grid_iterator operator++(int) {
                grid_iterator tmp(*this);
                operator++();
                return tmp;
            }

            array< size_t, D > &operator*() { return pos_; }

            bool operator==(const grid_iterator &other) const { return pos_ == other.pos_; }

            bool operator!=(const grid_iterator &other) const { return !operator==(other); }
        };

        grid_iterator begin() const { return grid_iterator{begin_, begin_, end_}; }
        grid_iterator end() const { return grid_iterator{end_, begin_, end_}; }

      private:
        array< size_t, D > begin_;
        array< size_t, D > end_;
    };

    /**
    * @brief Construct hypercube_view from a variadic sequence of ranges
    */
    //    template < typename... T /*, typename = all_integral< T... >*/ >
    //    GT_FUNCTION auto make_hypercube(T... r)
    //        // TODO do the transpose here
    //        GT_AUTO_RETURN((hypercube< typename std::common_type< T... >::type, sizeof...(T) >{r...}));
    //
    //    /**
    //    * @brief Construct hypercube_view from a sequence of brace-enclosed initializer lists
    //    * (where only the first two entries of each initializer lists are considered)
    //    */
    //    template < typename... T, typename = all_integral< T... > >
    //    GT_FUNCTION auto make_hypercube(std::initializer_list< T >... r)
    //        GT_AUTO_RETURN(make_hypercube(make_range(*r.begin(), *(r.begin() + 1))...));
    //
    //    template < typename T, size_t D >
    //    hypercube_view< T, D > make_hypercube_view(const hypercube< T, D > &hc) {
    //        return hypercube_view< T, D >(hc);
    //    }
    //
    //    template < typename... T, typename = all_integral< T... > >
    //    GT_FUNCTION auto make_hypercube_view(std::initializer_list< T >... r)
    //        GT_AUTO_RETURN(make_hypercube(make_range(*r.begin(), *(r.begin() + 1))...));

    //    template < typename... T >
    //    auto make_hypercube_view(T &&... t) GT_AUTO_RETURN(make_hypercube_view(make_hypercube(std::forward< T
    //    >(t)...)));

    //    template < typename IntT >
    //    using range_t = array< IntT, 2 >;
    //
    //    template < typename T >
    //    range_t< T > make_range(T left, T right) {
    //        return range_t< T >({left, right});
    //    }

    // TODO all the makers should actually create a range not a hypercube_view (because of reusability)

    /**
    * @brief Construct hypercube_view from array of pairs (ranges)
    */
    //    template < typename T, size_t Size, typename = all_integral< T > >
    //    GT_FUNCTION auto make_hypercube_view(const array< range< T >, Size > &range)
    //        GT_AUTO_RETURN((multi_iterator< T, Size >{range}));
}

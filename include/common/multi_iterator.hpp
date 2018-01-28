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
    namespace impl_ {
        /**
         * Recursively build an index within the iteration range:
         * - remove first element from (remaining) range...
         * - loop over range
         * - append index to "pos"
         *
         * 1 + sizeof...(Range) + sizeof...(Position) == #dimensions
         */
        // needed to be able to pass an empty range
        template < typename... Range >
        struct recursive_iterate;

        template < typename FirstFromRange, typename... Range >
        struct recursive_iterate< FirstFromRange, Range... > {
            GT_NV_EXEC_CHECK_DISABLE
            template < typename F, typename... Position >
            static GT_FUNCTION void apply(F f, FirstFromRange first_range, Range... range, Position... pos) {
                GRIDTOOLS_STATIC_ASSERT((is_all_integral< Position... >::value), GT_INTERNAL_ERROR);

                for (uint_t i = first_range.first; i < first_range.second; ++i) {
                    recursive_iterate< Range... >::apply(f, range..., pos..., i);
                }
            }
        };

        template <>
        struct recursive_iterate<> {
            GT_NV_EXEC_CHECK_DISABLE
            // position is fully build -> call functor
            template < typename F, typename... Position >
            static GT_FUNCTION void apply(F f, Position... pos) {
                f(pos...);
            }
        };

        /*
         * Same as recursive_iterate but with reduction
         * Note: All my versions were I tried to remove the code duplication with recursive_reduce were longer and more
         * complicated.
         */
        template < typename... Range >
        struct recursive_reduce;

        template < typename FirstFromRange, typename... Range >
        struct recursive_reduce< FirstFromRange, Range... > {
            GT_NV_EXEC_CHECK_DISABLE
            template < typename F, typename BinaryOp, typename InitT, typename... Position >
            static GT_FUNCTION InitT apply(
                F &&f, BinaryOp &&binop, InitT init, FirstFromRange first_range, Range... range, Position... pos) {
                GRIDTOOLS_STATIC_ASSERT((is_all_integral< Position... >::value), GT_INTERNAL_ERROR);

                InitT reduction = init;
                for (uint_t i = first_range.first; i < first_range.second; ++i) {
                    reduction =
                        binop(reduction,
                            recursive_reduce< Range... >::apply(
                                  std::forward< F >(f), std::forward< BinaryOp >(binop), init, range..., pos..., i));
                }
                return reduction;
            }
        };

        template <>
        struct recursive_reduce<> {
            GT_NV_EXEC_CHECK_DISABLE
            template < typename F, typename BinaryOp, typename InitT, typename... Position >
            static GT_FUNCTION auto apply(F &&f, BinaryOp &&binop, InitT init, Position... pos)
                GT_AUTO_RETURN(f(pos...));
        };

        template < typename T, size_t I >
        using make_type_sequence = T;

        // expand range array to variadic sequence
        template < typename T, typename F, size_t... Seq >
        GT_FUNCTION void iterate_array_to_variadic(
            F &&f, const gt_integer_sequence< size_t, Seq... > seq, const gridtools::array< T, sizeof...(Seq) > &a) {
            impl_::recursive_iterate< make_type_sequence< T, Seq >... >::apply(std::forward< F >(f), a[Seq]...);
        }

        // 0 dim range
        template < typename T, typename F >
        GT_FUNCTION void iterate_array_to_variadic(
            F &&f, const gt_integer_sequence< size_t > seq, const gridtools::array< T, 0 > &a) {
            // noop;
        }

        template < typename T, typename F, typename BinaryOp, typename InitT, size_t... Seq >
        GT_FUNCTION auto iterate_reduce_array_to_variadic(F &&f,
            BinaryOp &&binop,
            InitT init,
            const gt_integer_sequence< size_t, Seq... > seq,
            const gridtools::array< T, sizeof...(Seq) > &a)
            GT_AUTO_RETURN((impl_::recursive_reduce< make_type_sequence< T, Seq >... >::apply(
                std::forward< F >(f), std::forward< BinaryOp >(binop), init, a[Seq]...)));

        // 0 dim range
        template < typename T, typename F, typename BinaryOp, typename InitT >
        GT_FUNCTION auto iterate_reduce_array_to_variadic(F &&f,
            BinaryOp &&binop,
            InitT init,
            const gt_integer_sequence< size_t > seq,
            const gridtools::array< T, 0 > &a) GT_AUTO_RETURN(init);
    }

    /**
     * @brief Iterator over a hyper-cube
     * @tparam IntT integer type which is used to define the range
     * @tparam Dim dimension of the hyper-cube
     */
    template < typename IntT, size_t Dim >
    class multi_iterator {
        gridtools::array< gridtools::pair< IntT, IntT >, Dim > range_;

      public:
        /**
         * @brief Iteration space given by an array of pairs for each dimension such that [first,second),
         * i.e. interval is closed on the left and open on the right.
         */
        GT_FUNCTION multi_iterator(const gridtools::array< gridtools::pair< IntT, IntT >, Dim > &range)
            : range_(range){};

        /**
         * @brief Iterate over Functor F
         * @param f Functor which is called for each iteration point, i.e. needs to accept <Dim> arguments of type IntT.
         */
        template < typename F >
        GT_FUNCTION void iterate(F &&f) {
            impl_::iterate_array_to_variadic(std::forward< F >(f), make_gt_integer_sequence< size_t, Dim >(), range_);
        }

        /**
         * @brief Reduction over F with BinaryOp.
         * @param f Functor which is called for each iteration point, i.e. needs to accept <Dim> arguments of type IntT,
         * type of return value should match InitT
         * @param binop Binary functor which is used to do the reduction (should be associative as there is no guarantee
         * about the order)
         * @param init Inital value for the reduction
         */
        template < typename F, typename BinaryOp, typename InitT >
        GT_FUNCTION auto reduce(F &&f, BinaryOp &&binop, InitT init)
            GT_AUTO_RETURN(impl_::iterate_reduce_array_to_variadic(std::forward< F >(f),
                std::forward< BinaryOp >(binop),
                init,
                make_gt_integer_sequence< size_t, Dim >(),
                range_));

        GT_FUNCTION bool operator==(const multi_iterator< IntT, Dim > &rhs) const { return range_ == rhs.range_; }
        GT_FUNCTION bool operator!=(const multi_iterator< IntT, Dim > &rhs) const { return !operator==(rhs); }
    };

    /**
    * @brief Construct multi_iterator from array of pairs (ranges)
    */
    template < typename T, size_t Size, typename = all_integral< T > >
    GT_FUNCTION auto make_multi_iterator(const array< pair< T, T >, Size > &range)
        GT_AUTO_RETURN((multi_iterator< T, Size >{range}));

    /**
    * @brief Construct multi_iterator from array of integers which define the right bounds,
    * we assume iteration starts at 0.
    * -> convert to range which starts at 0 in each dim
    */
    template < typename T, size_t Size, typename = all_integral< T > >
    GT_FUNCTION multi_iterator< T, Size > make_multi_iterator(const array< T, Size > &range) {
        array< pair< T, T >, Size > pair_range;
        for (size_t i = 0; i < Size; ++i)
            pair_range[i] = {0, range[i]};
        return multi_iterator< T, Size >{pair_range};
    }

    /**
    * @brief Construct multi_iterator from a variadic sequence of pairs (the ranges)
    */
    template < typename... T, typename = all_integral< T... > >
    GT_FUNCTION auto make_multi_iterator(
        pair< T, T >... range) GT_AUTO_RETURN((multi_iterator< typename std::common_type< T... >::type, sizeof...(T) >{
        array< pair< typename std::common_type< T... >::type, typename std::common_type< T... >::type >, sizeof...(T) >{
            range...}}));

    /**
    * @brief Construct multi_iterator from a variadic sequence of integers (right bounds),
    * assuming iteration starts at 0
    */
    template < typename... T, typename = all_integral< T... > >
    GT_FUNCTION auto make_multi_iterator(T... range) GT_AUTO_RETURN(make_multi_iterator(pair< T, T >{0, range}...));

    /**
    * @brief Construct multi_iterator from a sequence of brace-enclosed initializer lists
    * (where only the first two entries of each initializer lists are considered)
    */
    template < typename... T, typename = all_integral< T... > >
    GT_FUNCTION auto make_multi_iterator(std::initializer_list< T >... range)
        GT_AUTO_RETURN(make_multi_iterator(pair< T, T >(*range.begin(), *(range.begin() + 1))...));

    /**
    * @brief Construct a 0D (empty) multi_iterator (useful if variadic arguments are forwarded to this function)
    */
    template < typename T = uint_t >
    GT_FUNCTION auto make_multi_iterator() GT_AUTO_RETURN((multi_iterator< T, 0 >{array< pair< T, T >, 0 >{}}));

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

    template < typename T >
    range< T > make_range(T b, T e) {
        return range< T >{b, e};
    }

    template < typename T, size_t D >
    class hypercube : public array< range< T >, D > {
      public:
        template < typename... T2, typename = all_integral< T2... > >
        GT_FUNCTION hypercube(const range< T2 > &... r)
            : array< range< T >, D >{r...} {}

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

    // should end be included?
    template < typename T, size_t D >
    class hypercube_view {
      public:
        hypercube_view(const hypercube< T, D > &range) : range_{range} {}

        struct grid_iterator {
            array< T, D > pos_;
            const hypercube< T, D > range_;

            grid_iterator(const array< T, D > &pos, const hypercube< T, D > &range) : pos_{pos}, range_{range} {}

            operator array< T, D >() const { return pos_; }

            grid_iterator &operator++() {
                for (T i = 0; i < D; ++i) {
                    T index = D - i - 1;
                    if (pos_[index] + 1 < range_[index].end()) {
                        pos_[index]++;
                        return *this;
                    } else {
                        pos_[index] = range_[index].begin();
                    }
                }
                // we reached the end
                for (T i = 0; i < D; ++i)
                    pos_[i] = range_[i].end();
                return *this;
            }

            grid_iterator operator++(int) {
                grid_iterator tmp(*this);
                operator++();
                return tmp;
            }

            array< T, D > &operator*() { return pos_; }

            bool operator==(const grid_iterator &other) const { return pos_ == other.pos_; }

            bool operator!=(const grid_iterator &other) const { return !operator==(other); }
        };

        grid_iterator begin() const { return grid_iterator{range_.begin(), range_}; }
        grid_iterator end() const { return grid_iterator{range_.end(), range_}; }

      private:
        hypercube< T, D > range_;
    };

    /**
    * @brief Construct hypercube_view from a variadic sequence of ranges
    */
    template < typename... T, typename = all_integral< T... > >
    GT_FUNCTION auto make_hypercube(range< T >... r)
        GT_AUTO_RETURN((hypercube< typename std::common_type< T... >::type, sizeof...(T) >{r...}));

    /**
    * @brief Construct hypercube_view from a sequence of brace-enclosed initializer lists
    * (where only the first two entries of each initializer lists are considered)
    */
    template < typename... T, typename = all_integral< T... > >
    GT_FUNCTION auto make_hypercube(std::initializer_list< T >... r)
        GT_AUTO_RETURN(make_hypercube(make_range(*r.begin(), *(r.begin() + 1))...));

    template < typename T, size_t D >
    hypercube_view< T, D > make_hypercube_view(const hypercube< T, D > &hc) {
        return hypercube_view< T, D >(hc);
    }

    template < typename... T, typename = all_integral< T... > >
    GT_FUNCTION auto make_hypercube_view(std::initializer_list< T >... r)
        GT_AUTO_RETURN(make_hypercube(make_range(*r.begin(), *(r.begin() + 1))...));

    template < typename... T >
    auto make_hypercube_view(T &&... t) GT_AUTO_RETURN(make_hypercube_view(make_hypercube(std::forward< T >(t)...)));

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

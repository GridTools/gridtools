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
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/include/count.hpp>
#include <boost/fusion/include/make_fused.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/filter_if.hpp>
#include "../../storage/storage-facility.hpp"
#include "../intermediate.hpp"

namespace gridtools {

    /**
     * @file
     * \brief this file contains the intermediate representation used in case of expandable parameters
     * */

    /**
       @brief the intermediate representation object

       The expandable parameters are long lists of storages on which the same stencils are applied,
       in a Single-Stencil-Multiple-Storage way. In order to avoid resource contemption usually
       it is convenient to split the execution in multiple stencil, each stencil operating on a chunk
       of the list. Say that we have an expandable parameters list of length 23, and a chunk size of
       4, we'll execute 5 stencil with a "vector width" of 4, and one stencil with a "vector width"
       of 3 (23%4).

       This object contains two unique pointers of @ref gridtools::intermediate type, one with a
       vector width
       corresponding to the expand factor defined by the user (4 in the previous example), and another
       one with a vector width of expand_factor%total_parameters (3 in the previous example).
       In case the total number of parameters is a multiple of the expand factor, the second
       intermediate object does not get instantiated.
     */
    template < typename Backend,
        typename MssDescriptorArray,
        typename Aggregator,
        typename Grid,
        typename ConditionalsSet,
        typename ReductionType,
        bool IsStateful,
        typename ExpandFactor >
    class intermediate_expand : public computation< Aggregator, ReductionType > {
        GRIDTOOLS_STATIC_ASSERT((is_backend< Backend >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(
            (is_meta_array_of< MssDescriptorArray, is_computation_token >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< Aggregator >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_expand_factor< ExpandFactor >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(
            (std::is_same< ReductionType, notype >::value), "Reduction is not allowed with expandable parameters");

        template < typename T >
        struct is_expandable : std::false_type {};

        template < ushort_t N, typename Storage, typename Location, bool Temporary >
        struct is_expandable< arg< N, Storage, Location, Temporary > > : is_vector< Storage > {};

        template < typename Arg, typename Storage >
        struct is_expandable< arg_storage_pair< Arg, Storage > > : is_expandable< Arg > {};

        template < typename T >
        using is_non_tmp_expandable = boost::mpl::and_< is_expandable< T >, boost::mpl::not_< is_tmp_arg< T > > >;

        template < typename T, uint_t N >
        struct convert_storage_type {
            using type = T;
        };
        template < typename T, uint_t N >
        struct convert_storage_type< std::vector< T >, N > {
            using type = data_store_field< T, N >;
        };

        template < uint_t N >
        struct convert_placeholder {
            template < typename >
            struct apply;
            template < uint_t I, typename S, typename L, bool T >
            struct apply< arg< I, S, L, T > > {
                using type = arg< I, typename convert_storage_type< S, N >::type, L, T >;
            };
        };

        template < uint N >
        using converted_aggregator_type = aggregator_type<
            typename boost::mpl::transform< typename Aggregator::placeholders_t, convert_placeholder< N > >::type >;

        template < uint N >
        using converted_intermediate = intermediate< Backend,
            MssDescriptorArray,
            converted_aggregator_type< N >,
            Grid,
            ConditionalsSet,
            ReductionType,
            IsStateful,
            N >;

        using base_t = typename intermediate_expand::computation;
        using base_t::m_domain;

        struct do_run {
            template < typename T >
            void operator()(T &obj) const {
                obj.run();
            }
        };
        struct do_steady {
            template < typename T >
            void operator()(T &obj) const {
                obj.steady();
            }
        };

        // private members
        const size_t m_size;
        const std::unique_ptr< converted_intermediate< ExpandFactor::value > > m_intermediate;
        const std::unique_ptr< converted_intermediate< 1 > > m_intermediate_remainder;

      public:
        /**
           @brief constructor

           Given expandable parameters with size N, creates other @ref gristools::expandable_parameters storages with
           dimension given by  @ref gridtools::expand_factor
         */
        template < typename Domain >
        intermediate_expand(Domain &&domain, Grid const &grid, ConditionalsSet const &conditionals)
            : base_t(std::forward< Domain >(domain)), m_size(get_expandable_size(m_domain)),
              m_intermediate(m_size >= ExpandFactor::value
                                 ? create_intermediate< ExpandFactor::value >(m_domain, grid, conditionals)
                                 : nullptr),
              m_intermediate_remainder(
                  m_size % ExpandFactor::value ? create_intermediate< 1 >(m_domain, grid, conditionals) : nullptr) {}

        /**
           @brief run the execution

           This method performs a run for the computation on each chunck of expandable parameters.
           Between two iterations it updates the @ref gridtools::aggregator_type, so that the storage
           pointers for the current chunck get substituted by the next chunk. At the end of the
           iterations, if the number of parameters is not multiple of the expand factor, the remaining
           chunck of storage pointers is consumed.
         */
        ReductionType run() override {
            for_each_computation(do_run{});
            return {};
        }

        /**
           @brief forwards to the m_intermediate member

           does not take into account the remainder kernel executed when the number of parameters is
           not multiple of the expand factor
         */
        std::string print_meter() override {
            assert(false);
            return {};
        }

        /**
           @brief forwards to the m_intermediate and m_intermediate_remainder members
         */
        void reset_meter() override {
            if (m_intermediate)
                m_intermediate->reset_meter();
            if (m_intermediate_remainder)
                m_intermediate_remainder->reset_meter();
        }

        double get_meter() override {
            double res = 0;
            if (m_intermediate)
                res += m_intermediate->get_meter();
            if (m_intermediate_remainder)
                res += m_intermediate_remainder->get_meter();
            return res;
        }

        /**
           @brief forward the call to the members
         */
        void ready() override {
            if (m_intermediate)
                m_intermediate->ready();
            if (m_intermediate_remainder)
                m_intermediate_remainder->ready();
        }

        /**
           @brief forward the call to the members
         */
        void steady() override { for_each_computation(do_steady{}); }

        /**
           @brief forward the call to the members
         */
        void finalize() override {
            // sync all data stores
            boost::fusion::for_each(m_domain.m_arg_storage_pair_list, _impl::sync_data_stores());
        }

      private:
        struct get_value_size_f {
            template < class T >
            size_t operator()(T const &t) const {
                return t.m_value.size();
            }
        };

        static size_t get_expandable_size(Aggregator const &src) {
            namespace f = boost::fusion;
            auto sizes =
                f::transform(f::filter_if< is_non_tmp_expandable< boost::mpl::_ > >(src.get_arg_storage_pairs()),
                    get_value_size_f());
            if (f::empty(sizes))
                return 0;
            size_t res = f::front(sizes);
            assert(f::count(sizes, res) == f::size(sizes) && "Non-tmp expandable parameters must have the same size");
            return res;
        }

        template < uint_t N >
        struct convert_storage {
            template < typename T >
            data_store_field< T, N > operator()(const std::vector< T > &src) const {
                assert(!src.empty());
                return {*src[0].get_storage_info_ptr()};
            }
            template < typename T >
            T operator()(const T &src) const {
                return src;
            }
        };

        template < uint_t N >
        struct convert_arg_storage_pair {
            template < typename Arg, typename Storage >
            arg_storage_pair< typename convert_placeholder< N >::template apply< Arg >::type,
                typename convert_storage_type< Storage, N >::type >
            operator()(arg_storage_pair< Arg, Storage > const &src) const {
                return {convert_storage< N >()(src.m_value)};
            }
            template < typename T >
            T operator()(const T &src) const {
                return src;
            }
        };

        template < typename T >
        struct maker {
            template < typename... Us >
            T operator()(Us &&... us) const {
                return {std::forward< Us >(us)...};
            }
        };

        template < uint_t N >
        static converted_aggregator_type< N > convert_aggregator(const Aggregator &src) {
            namespace f = boost::fusion;
            namespace m = boost::mpl;
            auto arg_storage_pairs =
                f::transform(f::filter_if< m::not_< is_tmp_arg< m::_ > > >(src.get_arg_storage_pairs()),
                    convert_arg_storage_pair< N >());
            return f::make_fused(maker< converted_aggregator_type< N > >())(std::move(arg_storage_pairs));
        }

        template < uint_t N >
        static converted_intermediate< N > *create_intermediate(
            const Aggregator &src, Grid const &grid, const ConditionalsSet &conditionals) {
            return new converted_intermediate< N >(convert_aggregator< N >(src), grid, conditionals);
        }

        template < typename F >
        void for_each_computation(const F &fun) {
            size_t i = 0;
            for (; m_size - i >= ExpandFactor::value; i += ExpandFactor::value) {
                assign(*m_intermediate, i);
                fun(*m_intermediate);
            }
            for (; i < m_size; ++i) {
                assign(*m_intermediate_remainder, i);
                fun(*m_intermediate_remainder);
            }
        }

        struct assign_arg_storage_pair {
            size_t m_offset;
            template < typename Src, typename Arg, typename Dst, uint_t N >
            void operator()(Src &src, arg_storage_pair< Arg, data_store_field< Dst, N > > &dst) const {
                assert(src.m_value.size() >= m_offset + N);
                for (uint_t i = 0; i != N; ++i)
                    dst.m_value.set(0, i, src.m_value[m_offset + i]);
            }
        };

        template < typename Dst >
        void assign(Dst &dst, size_t offset) const {
            namespace f = boost::fusion;
            namespace m = boost::mpl;
            using pred = is_non_tmp_expandable< m::_ >;
            f::for_each(f::zip(f::filter_if< pred >(m_domain.get_arg_storage_pairs()),
                            f::filter_if< pred >(dst.domain().get_arg_storage_pairs())),
                f::make_fused(assign_arg_storage_pair()));
        }
    };
}

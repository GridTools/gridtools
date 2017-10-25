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
#include <boost/utility.hpp>
#include <boost/fusion/include/count.hpp>

#include "../../common/vector_traits.hpp"
#include "intermediate_expand_metafunctions.hpp"

namespace gridtools {
    namespace _impl {

        /**
           @brief function is used to retrieve an aggregator type isntance when given a boost fusion
           vector containing data_stores, etc. (base case)
           @tparam AggregatorType the result type (aggregator_type)
           @tparam DataStoreFieldVec the fusion vector type
           @tparam DataStoreFields variadic pack of data_store_fields
           @param dsf_vec fusion vector containing the data_store_fields
           @param dsf variadic list of data_store_fields
        */
        template < typename AggregatorType, typename DataStoreFieldVec, typename... DataStoreFields >
        typename boost::enable_if_c< sizeof...(DataStoreFields) == boost::mpl::size< DataStoreFieldVec >::value,
            AggregatorType >::type
        make_aggregator(DataStoreFieldVec &dsf_vec, DataStoreFields &... dsf) {
            return {dsf...};
        }

        /**
           @brief function is used to retrieve an aggregator type isntance when given a boost fusion
           vector containing data_stores, etc. (step case)
           @tparam AggregatorType the result type (aggregator_type)
           @tparam DataStoreFieldVec the fusion vector type
           @tparam DataStoreFields variadic pack of data_store_fields
           @param dsf_vec fusion vector containing the data_store_fields
           @param dsf variadic list of data_store_fields
        */
        template < typename AggregatorType, typename DataStoreFieldVec, typename... DataStoreFields >
        typename boost::enable_if_c< sizeof...(DataStoreFields) < boost::mpl::size< DataStoreFieldVec >::value,
            AggregatorType >::type
        make_aggregator(DataStoreFieldVec &dsf_vec, DataStoreFields &... dsf) {
            return make_aggregator< AggregatorType >(dsf_vec,
                dsf...,
                boost::fusion::deref(
                    boost::fusion::advance_c< sizeof...(DataStoreFields) >(boost::fusion::begin(dsf_vec)))
                    .m_value);
        }

        template < typename T >
        struct is_expandable : std::false_type {};

        template < ushort_t N, typename Storage, typename Location, bool Temporary >
        struct is_expandable< arg< N, Storage, Location, Temporary > > : is_vector< Storage > {};

        template < typename Arg, typename Storage >
        struct is_expandable< arg_storage_pair< Arg, Storage > > : is_vector< Storage > {};

        // generates an mpl::vector of the original (large) expandable parameters storage types
        template < typename Aggregator >
        using expandable_params_t =
            typename boost::mpl::fold< typename Aggregator::placeholders_t,
                boost::mpl::vector0<>,
                boost::mpl::if_< boost::mpl::and_< is_expandable< boost::mpl::_2 >,
                                     boost::mpl::not_< is_tmp_arg< boost::mpl::_2 > > >,
                                           boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 >,
                                           boost::mpl::_1 > >::type;

        /**
           @brief functor used to initialize the storage in a boost::fusion::vector full an
           instance of gridtools::aggregator_type
        */
        template < typename DomainFull, typename Vec >
        struct initialize_storage {
          private:
            DomainFull const &m_dom_full;
            Vec &m_vec_to;

          public:
            initialize_storage(DomainFull const &dom_full_, Vec &vec_to_) : m_dom_full(dom_full_), m_vec_to(vec_to_) {}

            /**
               @brief initialize the storage vector, specialization for the expandable args
             */
            template < ushort_t ID, typename T, typename L >
            void operator()(arg< ID, std::vector< T >, L, false >) {
                typedef arg< ID, std::vector< T >, L, false > placeholder_t;
                typedef typename boost::mpl::at_c< Vec, ID >::type arg_storage_pair_t;
                typedef typename arg_storage_pair_t::storage_t data_store_field_t;
                const auto expandable_param = m_dom_full.template get_arg_storage_pair< placeholder_t >().m_value;
                auto val = data_store_field_t{*expandable_param[0].get_storage_info_ptr()};
                // fill in the first bunch of ptrs
                for (unsigned i = 0; i < data_store_field_t::num_of_storages; ++i) {
                    val.set(0, i, expandable_param[i]);
                }
                boost::fusion::at< static_ushort< ID > >(m_vec_to) = {val};
            }

            template < ushort_t ID, typename T, typename L >
            void operator()(arg< ID, std::vector< T >, L, true >) {}

            /**
               @brief initialize the storage vector, specialization for the normal args
             */
            template < ushort_t ID, typename Storage, typename Location, bool Temporary >
            void operator()(arg< ID, Storage, Location, Temporary >) {
                // copy the gridtools pointer
                boost::fusion::at< static_ushort< ID > >(m_vec_to) =
                    m_dom_full.template get_arg_storage_pair< arg< ID, Storage, Location, Temporary > >();
            }
        };

        //        struct create_arg {
        //            template < typename T, typename ExpandFactor >
        //            struct apply {
        //                typedef data_store_field< typename get_storage_from_arg< T >::type, ExpandFactor::value >
        //                exp_param_t;
        //                typedef arg< arg_index< T >::value, exp_param_t, typename T::location_t, T::is_temporary >
        //                type;
        //            };
        //        };

        //        template <typename Aggregator, uint_t N>
        //        using expand_arg_list_t = typename boost::mpl::fold<typename Aggregator::placeholders_t,
        //                boost::mpl::vector0<>, boost::mpl::push_back< boost::mpl::_1, boost::mpl::if_<
        //                _impl::is_expandable<
        //                        boost::mpl::_2 >, typename convert_arg_type< boost::mpl::_2, N >::type,
        //                                boost::mpl::_2 > > >::type expand_arg_list;

        /*
         *                 expand_vec_t expand_vec;
                        // initialize the storage list objects, whithout allocating the storage for the data snapshots
                        boost::mpl::for_each< typename Aggregator::placeholders_t >(
                                _impl::initialize_storage< Aggregator, expand_vec_t >(domain, expand_vec, m_size));

                        auto non_tmp_expand_vec =
                                boost::fusion::filter_if< boost::mpl::not_< is_arg_storage_pair_to_tmp< boost::mpl::_ >
         > >(expand_vec);

         */

        struct get_value_size_f {
            using result_type = size_t;
            template < class T >
            size_t operator()(T const &t) const {
                return t.m_value.size();
            }
        };

        template < class Aggregator >
        size_t get_expandable_size(Aggregator const &agg) {
            namespace f = boost::fusion;
            namespace m = boost::mpl;
            using m::_;
            using is_expandable_t = m::and_< is_expandable< _ >, m::not_< is_tmp_arg< _ > > >;
            auto sizes = f::transform(f::filter_if< is_expandable_t >(agg.get_arg_storage_pairs()), get_value_size_f());
            if (f::empty(sizes))
                return 0;
            size_t res = f::front(sizes);
            assert(f::count(sizes, res) == f::size(sizes) && "Non-tmp expandable parameters must have the same size");
            return res;
        }

        //        template < typename Src, typename Dst >
        //        void initialize_storage(Src const &src, Dst& dst) {
        //            boost::mpl::for_each<Src::non_tmp_placeholders_t>(initialize_storage_f<Src, Dst>{src, dst});
        //
        //        };

        /**
           @brief functor used to assign the next chunk of storage pointers
        */
        template < typename ExpandFactor, typename DomainFull, typename DomainChunk >
        struct assign_expandable_params_f {
            DomainFull const &m_dom_full;
            DomainChunk &m_dom_chunk;
            uint_t m_idx;

            template < ushort_t ID, typename T, typename L >
            void operator()(arg< ID, std::vector< T >, L, true >) {}

            template < ushort_t ID, typename T, typename L >
            void operator()(arg< ID, std::vector< T >, L, false >) {
                // the vector of pointers
                typedef arg< ID, std::vector< T >, L, false > placeholder_t;
                const auto &src = m_dom_full.template get_arg_storage_pair< placeholder_t >().m_value;
                auto &dst = boost::fusion::at< static_ushort< ID > >(m_dom_chunk.m_arg_storage_pair_list).m_value;
                for (unsigned i = 0; i < ExpandFactor::value; ++i)
                    dst.set(0, i, src[m_idx + i]);
            }
        };

        template < typename Factor, typename Src, typename Dst >
        void assign_expandable_params(Src const &src, Dst &dst, uint_t idx) {
            boost::mpl::for_each< expandable_params_t< Src > >(
                assign_expandable_params_f< Factor, Src, Dst >{src, dst, idx});
        }

    } // namespace _impl
} // namespace gridtools

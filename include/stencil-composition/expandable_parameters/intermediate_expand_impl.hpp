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

#include "../../common/vector_traits.hpp"

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
            AggregatorType * >::type
        make_aggregator(DataStoreFieldVec &dsf_vec, DataStoreFields &... dsf) {
            return new AggregatorType(dsf...);
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
            AggregatorType * >::type
        make_aggregator(DataStoreFieldVec &dsf_vec, DataStoreFields &... dsf) {
            return make_aggregator< AggregatorType >(dsf_vec,
                dsf...,
                *(boost::fusion::deref(boost::fusion::advance_c< sizeof...(DataStoreFields) >(boost::fusion::begin(
                                           dsf_vec))).ptr));
        }

        /**
           @brief functor used to initialize the storage in a boost::fusion::vector full an
           instance of gridtools::aggregator_type
        */
        template < typename DomainFull, typename Vec >
        struct initialize_storage {
          private:
            DomainFull const &m_dom_full;
            Vec &m_vec_to;
            bool m_called;
            ushort_t &m_size;

          public:
            initialize_storage(DomainFull const &dom_full_, Vec &vec_to_, ushort_t &size)
                : m_dom_full(dom_full_), m_vec_to(vec_to_), m_called(false), m_size(size) {}

            /**
               @brief initialize the storage vector, specialization for the expandable args
             */
            template < ushort_t ID, typename T, typename L >
            void operator()(arg< ID, std::vector< T >, L, false >) {
                typedef arg< ID, std::vector< T >, L, false > placeholder_t;
                typedef typename boost::mpl::at_c< Vec, ID >::type arg_storage_pair_t;
                typedef typename arg_storage_pair_t::storage_t data_store_field_t;
                const auto expandable_param =
                    (*(m_dom_full.template get_arg_storage_pair< placeholder_t, placeholder_t >()).ptr);
                data_store_field_t *ptr = new data_store_field_t(*(expandable_param[0].get_storage_info_ptr()));
                // fill in the first bunch of ptrs
                for (unsigned i = 0; i < data_store_field_t::size; ++i) {
                    ptr->set(0, i, expandable_param[i]);
                }
                boost::fusion::at< static_ushort< ID > >(m_vec_to) =
                    arg_storage_pair_t(static_cast< data_store_field_t * >(ptr));
                // the lines below are checking if the expandable params are all of the same size
                if (m_called) {
                    assert(
                        m_size == expandable_param.size() && "Non-tmp expandable parameters must have the same size");
                }
                m_called = true;
                m_size = expandable_param.size();
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

        /**
           @brief functor used to delete the storages containing the chunk of pointers
        */
        template < typename Vec >
        struct delete_storage {

          private:
            Vec &m_vec_to;

          public:
            delete_storage(Vec &vec_to_) : m_vec_to(vec_to_) {}

            /**
               @brief delete the non temporary data store fields
             */
            template < ushort_t ID, typename T, typename L >
            void operator()(arg< ID, std::vector< T >, L, false >) {
                delete (boost::fusion::at< static_ushort< ID > >(m_vec_to).ptr.get());
            }

            template < ushort_t ID, typename Storage, typename Location, bool Temporary >
            void operator()(arg< ID, Storage, Location, Temporary >) {}
        };

        /**
           @brief functor used to assign the next chunk of storage pointers
        */
        template < typename ExpandFactor, typename Backend, typename DomainFull, typename DomainChunk >
        struct assign_expandable_params {

          private:
            DomainFull const &m_dom_full;
            DomainChunk &m_dom_chunk;
            uint_t const &m_idx;

          public:
            assign_expandable_params(DomainFull const &dom_full_, DomainChunk &dom_chunk_, uint_t const &i_)
                : m_dom_full(dom_full_), m_dom_chunk(dom_chunk_), m_idx(i_) {}

            template < ushort_t ID, typename T, typename L >
            void operator()(arg< ID, std::vector< T >, L, true >) {}

            template < ushort_t ID, typename T, typename L >
            void operator()(arg< ID, std::vector< T >, L, false >) {
                // the vector of pointers
                typedef arg< ID, std::vector< T >, L, false > placeholder_t;
                pointer< std::vector< T > > const &ptr_full_ =
                    m_dom_full.template get_arg_storage_pair< placeholder_t, placeholder_t >().ptr;

                auto ptr_chunk_ = boost::fusion::at< static_ushort< ID > >(m_dom_chunk.m_arg_storage_pair_list);
#ifndef NDEBUG
                if (!ptr_chunk_.ptr.get() || !ptr_full_.get()) {
                    printf("The storage pointer is already null. Did you call finalize too early?");
                    assert(false);
                }
#endif
                for (unsigned i = 0; i < ExpandFactor::value; ++i) {
                    (*(ptr_chunk_.ptr)).set(0, i, (*ptr_full_)[m_idx + i]);
                }
            }
        };

    } // namespace _impl
} // namespace gridtools

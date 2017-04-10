/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/filter_if.hpp>

#include <storage-facility.hpp>

#include "../intermediate.hpp"

#include "intermediate_expand_impl.hpp"
#include "intermediate_expand_metafunctions.hpp"

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
        typename DomainType,
        typename Grid,
        typename ConditionalsSet,
        typename ReductionType,
        bool IsStateful,
        typename ExpandFactor >
    struct intermediate_expand : public computation< ReductionType > {
        GRIDTOOLS_STATIC_ASSERT((is_backend< Backend >::value), "wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_meta_array_of< MssDescriptorArray, is_computation_token >::value), "wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< DomainType >::value), "wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_expand_factor< ExpandFactor >::value), "wrong type");

        // create an mpl vector of @ref gridtools::arg, substituting the large
        // expandable parameters list with a chunk
        typedef typename boost::mpl::fold<
            typename DomainType::placeholders_t,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1,
                boost::mpl::if_< _impl::is_expandable_arg< boost::mpl::_2 >,
                                       typename _impl::create_arg::template apply< boost::mpl::_2, ExpandFactor >,
                                       boost::mpl::_2 > > >::type expand_arg_list;

        // create an mpl vector of @ref gridtools::arg, substituting the large
        // expandable parameters list with a chunk
        typedef typename boost::mpl::fold<
            typename DomainType::placeholders_t,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1,
                boost::mpl::if_< _impl::is_expandable_arg< boost::mpl::_2 >,
                                       typename _impl::create_arg::template apply< boost::mpl::_2, expand_factor< 1 > >,
                                       boost::mpl::_2 > > >::type expand_arg_list_remainder;

        // generates an mpl::vector of the original (large) expandable parameters storage types
        typedef typename boost::mpl::fold< typename DomainType::placeholders_t,
            boost::mpl::vector0<>,
            boost::mpl::if_< boost::mpl::and_< _impl::is_expandable_arg< boost::mpl::_2 >,
                                 boost::mpl::not_< is_tmp_arg< boost::mpl::_2 > > >,
                                               boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 >,
                                               boost::mpl::_1 > >::type expandable_params_t;

        // typedef to the intermediate type associated with the vector length of ExpandFactor::value
        typedef intermediate< Backend,
            MssDescriptorArray,
            aggregator_type< expand_arg_list >,
            Grid,
            ConditionalsSet,
            ReductionType,
            IsStateful,
            ExpandFactor::value > intermediate_t;

        // typedef to the intermediate type associated with the vector length of s_size%ExpandFactor::value
        typedef intermediate< Backend,
            MssDescriptorArray,
            aggregator_type< expand_arg_list_remainder >,
            Grid,
            ConditionalsSet,
            ReductionType,
            IsStateful,
            1 > intermediate_remainder_t;

      private:
        // private members
        DomainType const &m_domain_full;
        std::unique_ptr< aggregator_type< expand_arg_list > > m_domain_chunk;
        std::unique_ptr< aggregator_type< expand_arg_list_remainder > > m_domain_chunk_remainder;
        std::unique_ptr< intermediate_t > m_intermediate;
        std::unique_ptr< intermediate_remainder_t > m_intermediate_remainder;
        ushort_t m_size;

        // fusion vector of storage lists
        typedef aggregator_type< expand_arg_list > aggregator_t;
        typedef aggregator_type< expand_arg_list_remainder > aggregator_remainder_t;
        typedef typename aggregator_t::arg_storage_pair_fusion_list_t expand_vec_t;
        typedef typename aggregator_remainder_t::arg_storage_pair_fusion_list_t expand_vec_remainder_t;
        expand_vec_t expand_vec;
        expand_vec_remainder_t expand_vec_remainder;

      public:
        /**
           @brief constructor

           Given expandable parameters with size N, creates other @ref gristools::expandable_parameters storages with
           dimension given by  @ref gridtools::expand_factor
         */
        intermediate_expand(DomainType &domain, Grid const &grid, ConditionalsSet conditionals_)
            : m_domain_full(domain), m_domain_chunk(), m_domain_chunk_remainder(), m_intermediate(),
              m_intermediate_remainder(), m_size(0) {

            // initialize the storage list objects, whithout allocating the storage for the data snapshots
            boost::mpl::for_each< typename DomainType::placeholders_t >(
                _impl::initialize_storage< DomainType, expand_vec_t >(domain, expand_vec, m_size));

            auto non_tmp_expand_vec =
                boost::fusion::filter_if< boost::mpl::not_< is_arg_storage_pair_to_tmp< boost::mpl::_ > > >(expand_vec);
            m_domain_chunk.reset(_impl::get_aggregator< aggregator_t >(non_tmp_expand_vec));

            if (m_size >= ExpandFactor::value)
                m_intermediate.reset(new intermediate_t(*m_domain_chunk, grid, conditionals_));

            if (m_size % ExpandFactor::value) {
                boost::mpl::for_each< typename DomainType::placeholders_t >(
                    _impl::initialize_storage< DomainType, expand_vec_remainder_t >(
                        domain, expand_vec_remainder, m_size));
                auto non_tmp_expand_vec_remainder =
                    boost::fusion::filter_if< boost::mpl::not_< is_arg_storage_pair_to_tmp< boost::mpl::_ > > >(
                        expand_vec_remainder);
                m_domain_chunk_remainder.reset(
                    _impl::get_aggregator< aggregator_remainder_t >(non_tmp_expand_vec_remainder));
                m_intermediate_remainder.reset(
                    new intermediate_remainder_t(*m_domain_chunk_remainder, grid, conditionals_));
            }
        }

        /**
           @brief Method to reassign the storage pointers in the aggregator_type

           @param args the arguments are storages (data stores, data store fields, etc.)
           see @ref gridtools::test_domain_reassign for reference
         */
        template < typename... DataStores >
        void reassign(DataStores &... stores) {
            if (m_size >= ExpandFactor::value)
                m_intermediate->reassign(stores...);
            if (m_size % ExpandFactor::value)
                m_intermediate_remainder->reassign(stores...);
        }

        /**
           @brief run the execution

           This method performs a run for the computation on each chunck of expandable parameters.
           Between two iterations it updates the @ref gridtools::aggregator_type, so that the storage
           pointers for the current chunck get substituted by the next chunk. At the end of the
           iterations, if the number of parameters is not multiple of the expand factor, the remaining
           chunck of storage pointers is consumed.
         */
        virtual auto run() -> decltype(m_intermediate_remainder->run()) {
            GRIDTOOLS_STATIC_ASSERT((boost::is_same< decltype(m_intermediate_remainder->run()), notype >::value),
                "Reduction is not allowed with expandable parameters");
            // the expand factor must be smaller than the total size of the expandable parameters list
            for (uint_t i = 0; i < m_size - m_size % ExpandFactor::value; i += ExpandFactor::value) {
                boost::mpl::for_each< expandable_params_t >(_impl::assign_expandable_params< ExpandFactor,
                    Backend,
                    DomainType,
                    aggregator_type< expand_arg_list > >(m_domain_full, *m_domain_chunk, i));
                m_intermediate->run();
            }
            for (uint_t i = 0; i < m_size % ExpandFactor::value; ++i) {
                boost::mpl::for_each< expandable_params_t >(_impl::assign_expandable_params< boost::mpl::int_< 1 >,
                    Backend,
                    DomainType,
                    aggregator_type< expand_arg_list_remainder > >(
                    m_domain_full, *m_domain_chunk_remainder, m_size - m_size % ExpandFactor::value + i));
                m_intermediate_remainder->run();
            }
            return 0.; // reduction disabled
        }

        /**
           @brief forwards to the m_intermediate member

           does not take into account the remainder kernel executed when the number of parameters is
           not multiple of the expand factor
         */
        virtual std::string print_meter() { assert(false); }

        /**
           @brief forwards to the m_intermediate and m_intermediate_remainder members
         */
        virtual void reset_meter() {
            if (m_size >= ExpandFactor::value)
                m_intermediate->reset_meter();
            if (m_size % ExpandFactor::value)
                m_intermediate_remainder->reset_meter();
        }

        virtual double get_meter() {
            return m_intermediate->get_meter() +
                   ((m_size % ExpandFactor::value) ? m_intermediate_remainder->get_meter() : 0.);
        }

        /**
           @brief forward the call to the members
         */
        virtual void ready() {
            if (m_size >= ExpandFactor::value)
                m_intermediate->ready();

            if (m_size % ExpandFactor::value)
                m_intermediate_remainder->ready();
        }

        /**
           @brief forward the call to the members
         */
        virtual void steady() {
            for (uint_t i = 0; i < m_size - m_size % ExpandFactor::value; i += ExpandFactor::value) {
                boost::mpl::for_each< expandable_params_t >(_impl::assign_expandable_params< ExpandFactor,
                    Backend,
                    DomainType,
                    aggregator_type< expand_arg_list > >(m_domain_full, *m_domain_chunk, i));
                m_intermediate->steady();
            }
            for (uint_t i = 0; i < m_size % ExpandFactor::value; ++i) {
                boost::mpl::for_each< expandable_params_t >(_impl::assign_expandable_params< boost::mpl::int_< 1 >,
                    Backend,
                    DomainType,
                    aggregator_type< expand_arg_list_remainder > >(
                    m_domain_full, *m_domain_chunk_remainder, m_size - m_size % ExpandFactor::value + i));
                m_intermediate_remainder->steady();
            }
        }

        /**
           @brief forward the call to the members
         */
        virtual void finalize() {
            // sync all data stores
            boost::fusion::for_each(m_domain_full.m_arg_storage_pair_list, _impl::sync_data_stores());
            if (m_size >= ExpandFactor::value)
                m_intermediate->finalize();

            if (m_size % ExpandFactor::value)
                m_intermediate_remainder->finalize();

            // copy pointers back
            boost::mpl::for_each< typename DomainType::placeholders_t >(
                _impl::delete_storage< expand_vec_t >(expand_vec));
            boost::mpl::for_each< typename DomainType::placeholders_t >(
                _impl::delete_storage< expand_vec_remainder_t >(expand_vec_remainder));
        }
    };
}

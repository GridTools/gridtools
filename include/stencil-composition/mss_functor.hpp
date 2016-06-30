/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
/*
 * mss_functor.h
 *
 *  Created on: Mar 5, 2015
 *      Author: carlosos
 */

#pragma once

#include "common/meta_array.hpp"
#include "mss_metafunctions.hpp"
#include "mss_local_domain.hpp"
#include "mss.hpp"
#include "axis.hpp"
#include "mss_components_metafunctions.hpp"
#include "run_functor_arguments.hpp"

namespace gridtools {

#ifdef CXX11_ENABLED
    template < typename MssComponents, typename IntervalsMapSeq >
    struct mss_components_functors_return_type {
        GRIDTOOLS_STATIC_ASSERT((is_mss_components< MssComponents >::value), "Internal Error");

        template < typename EsfFunction, typename IntervalPair >
        struct functor_return_type {
            typedef typename boost::mpl::second< IntervalPair >::type interval_t;

            GRIDTOOLS_STATIC_ASSERT((has_do< EsfFunction, interval_t >::type::value),
                "Error: Do method does not contain signature with specified interval");
            using rtype = decltype(EsfFunction::Do(int(), interval_t()));
            typedef typename boost::mpl::if_< boost::is_same< void, rtype >, notype, rtype >::type type;
        };

        template < typename EsfSequence >
        struct esfs_functor_return_type {
            GRIDTOOLS_STATIC_ASSERT(
                (boost::mpl::size< EsfSequence >::value == 1), "Error: Reductions can have only one esf");
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< IntervalsMapSeq >::value == 1), "Error");

            typedef typename boost::mpl::front< IntervalsMapSeq >::type intervals_map_t;

            typedef typename boost::mpl::front< EsfSequence >::type::esf_function esf_function_t;
            typedef typename boost::mpl::fold< intervals_map_t,
                boost::mpl::set0<>,
                boost::mpl::insert< boost::mpl::_1, functor_return_type< esf_function_t, boost::mpl::_2 > > >::type
                return_type_seq;

            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< return_type_seq >::value == 1),
                "All overloaded Do methods of a reduction functor should return the same type");
            typedef typename boost::mpl::front< return_type_seq >::type type;
        };

        typedef typename boost::mpl::eval_if< typename MssComponents::mss_descriptor_t::is_reduction_t,
            esfs_functor_return_type< typename MssComponents::mss_descriptor_t::esf_sequence_t >,
            boost::mpl::identity< notype > >::type type;
    };
#endif

    /**
     * @brief functor that executes all the functors contained within the mss
     * @tparam TMssArray meta array containing all the mss descriptors
     * @tparam Grid grid
     * @tparam MssLocalDomainArray sequence of mss local domain (each contains the local domain list of a mss)
     * @tparam BackendIds ids of backends
     */
    template < typename MssComponentsArray,
        typename Grid,
        typename MssLocalDomainArray,
        typename BackendIds,
        typename ReductionData >
    struct mss_functor {
        GRIDTOOLS_STATIC_ASSERT(
            (is_sequence_of< MssLocalDomainArray, is_mss_local_domain >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT(
            (is_meta_array_of< MssComponentsArray, is_mss_components >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_backend_ids< BackendIds >::value), "Error");
        GRIDTOOLS_STATIC_ASSERT((is_reduction_data< ReductionData >::value), "Error");

#ifdef CXX11_ENABLED
        template < typename MssComponents, typename FunctorsMap >
        struct check_reduction_types {
            GRIDTOOLS_STATIC_ASSERT((is_mss_components< MssComponents >::value), "Error");
            // extract the type derived from the return type of the user functors of the reduction
            typedef typename mss_components_functors_return_type< MssComponents, FunctorsMap >::type reduction_t;

            // extract the type derived from the type specified by the user in the API when an initial value is passed
            typedef typename boost::mpl::eval_if< mss_components_is_reduction< MssComponents >,
                reduction_descriptor_type< typename MssComponents::mss_descriptor_t >,
                boost::mpl::identity< notype > >::type functor_return_t;

            // verify that both types are the same
            GRIDTOOLS_STATIC_ASSERT((boost::is_same< functor_return_t, reduction_t >::value),
                "Return type of reduction functors does not match the type of initialized value of the reduction");
            // verify that the deduced types are the same as the same passed from intermediate via the ReductionData
            typedef typename boost::mpl::eval_if< mss_components_is_reduction< MssComponents >,
                typename boost::is_same< functor_return_t, typename ReductionData::reduction_type_t >::type,
                boost::mpl::true_ >::type type;

            GRIDTOOLS_STATIC_ASSERT((type::value), "Error");
        };
#endif

      private:
        MssLocalDomainArray &m_local_domain_lists;
        const Grid &m_grid;
        ReductionData &m_reduction_data;
        const uint_t m_block_idx, m_block_idy;

      public:
        mss_functor(MssLocalDomainArray &local_domain_lists,
            const Grid &grid,
            ReductionData &reduction_data,
            const int block_idx,
            const int block_idy)
            : m_local_domain_lists(local_domain_lists), m_grid(grid), m_reduction_data(reduction_data),
              m_block_idx(block_idx), m_block_idy(block_idy) {}

        template < typename T1, typename T2, typename Seq, typename NextSeq >
        struct condition_for_async {

            typedef typename boost::mpl::and_< typename boost::mpl::at< Seq, T2 >::type,
                typename boost::mpl::at< NextSeq, T2 >::type >::type type;
        };

        /**
         * \brief given the index of a functor in the functors list ,it calls a kernel on the GPU executing the
         * operations defined on that functor.
         */
        template < typename Index >
        void operator()(Index const &) const {
            typedef typename boost::fusion::result_of::value_at< MssLocalDomainArray, Index >::type mss_local_domain_t;
            GRIDTOOLS_STATIC_ASSERT((is_mss_local_domain< mss_local_domain_t >::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT(
                (Index::value < boost::mpl::size< typename MssComponentsArray::elements >::value), "Internal Error");
            typedef typename boost::mpl::at< typename MssComponentsArray::elements, Index >::type mss_components_t;

            typedef typename mss_local_domain_list< mss_local_domain_t >::type local_domain_list_t;
            typedef typename mss_local_domain_esf_args_map< mss_local_domain_t >::type local_domain_esf_args_map_t;

            GRIDTOOLS_STATIC_ASSERT(
                (boost::mpl::size< local_domain_list_t >::value == 1), "Internal Error: wrong size");
            typedef typename boost::mpl::back< local_domain_list_t >::type local_domain_t;
            local_domain_list_t &local_domain_list =
                (local_domain_list_t &)boost::fusion::at< Index >(m_local_domain_lists).local_domain_list;
            local_domain_t &local_domain =
                (local_domain_t &)boost::fusion::at< boost::mpl::int_< 0 > >(local_domain_list);

            typedef typename mss_components_t::execution_engine_t ExecutionEngine;

            typedef typename mss_loop_intervals< mss_components_t, Grid >::type
                LoopIntervals; // List of intervals on which functors are defined
            // wrapping all the template arguments in a single container
            typedef typename boost::mpl::if_<
                typename boost::mpl::bool_< ExecutionEngine::type::iteration == enumtype::forward >::type,
                LoopIntervals,
                typename boost::mpl::reverse< LoopIntervals >::type >::type oriented_loop_intervals_t;
            // List of functors to execute (in order)
            typedef typename mss_components_t::functors_list_t functors_list_t;
            // sequence of esf descriptors contained in this mss
            typedef typename mss_components_t::linear_esf_t esf_sequence_t;
            // computed extent sizes to know where to compute functot at<i>
            typedef typename mss_components_t::extent_sizes_t extent_sizes;
            // Map between interval and actual arguments to pass to Do methods
            typedef typename mss_functor_do_method_lookup_maps< mss_components_t, Grid >::type functors_map_t;

// we check that the return types of the reduction user functor matcheds the value provided in the initial value
// extracted from the make_reduction api (only available for CXX11)
#ifdef CXX11_ENABLED
            typedef typename check_reduction_types< mss_components_t, functors_map_t >::type reduction_type_check_t;
#endif
            typedef backend_traits_from_id< BackendIds::s_backend_id > backend_traits_t;

            typedef typename backend_traits_t::template get_block_size< BackendIds::s_strategy_id >::type block_size_t;
            // compute the struct with all the type arguments for the run functor

            typedef typename sequence_of_is_independent_esf< typename mss_components_t::mss_descriptor_t >::type
                sequence_of_is_independent_t;

            /** generates the map of stating which esf has to be synchronized

                this is what the following metafunction does:

                - sets the last boolean of the vector to TRUE (never need to sync the last ESF)
                - loops over the inner linearized ESFs starting from 0, excluding the last one
                - if the next ESF is not independent, then this one needs to be synced, otherwise
                it is not synced
                , i.e.: if there are 2 independent ESFs in a row, the first one does not need the sync

                NOTE: this could be avoided if our systax at the user level was different, i.e. instead of having
                  make_mss(
                     make_stage( esf_0),
                     make_stage( esf_1),
                     make_independent(esf_2, esf_3, esf_4, esf_5));

                  we had e.g.

                  make_mss(
                     make_stage( esf_0 ),
                     make_stage( esf_1 ),
                     make_stage( esf_2 ),
                     make_independent(esf_3, esf_4, esf_5));

                     (which is less intuitive though)

             */

            typedef
                typename boost::mpl::fold< boost::mpl::range_c< int, 1, boost::mpl::size< esf_sequence_t >::value >,
                    boost::mpl::vector0<>,
                    boost::mpl::push_back< boost::mpl::_1,
                                               boost::mpl::at< sequence_of_is_independent_t, boost::mpl::_2 > > >::type
                    next_thing;

            typedef typename boost::mpl::fold<
                boost::mpl::range_c< int, 0, boost::mpl::size< next_thing >::value >,
                boost::mpl::map<>,
                boost::mpl::if_<
                    condition_for_async< boost::mpl::_1, boost::mpl::_2, sequence_of_is_independent_t, next_thing >,
                    boost::mpl::insert< boost::mpl::_1,
                        boost::mpl::pair< boost::mpl::at< functors_list_t, boost::mpl::_2 >, boost::mpl::true_ > >,
                    boost::mpl::insert< boost::mpl::_1,
                        boost::mpl::pair< boost::mpl::at< functors_list_t, boost::mpl::_2 >,
                                            boost::mpl::false_ > > > >::type async_esf_map_tmp_t;

            // insert true for the last esf
            typedef typename boost::mpl::insert< async_esf_map_tmp_t,
                boost::mpl::pair< typename boost::mpl::at_c< functors_list_t,
                                      boost::mpl::size< next_thing >::value >::type,
                                                     boost::mpl::true_ > >::type async_esf_map_t;

            // perform some checks concerning the reduction types
            typedef run_functor_arguments< BackendIds,
                block_size_t,
                block_size_t,
                functors_list_t,
                esf_sequence_t,
                local_domain_esf_args_map_t,
                oriented_loop_intervals_t,
                functors_map_t,
                extent_sizes,
                local_domain_t,
                typename mss_components_t::cache_sequence_t,
                async_esf_map_t,
                Grid,
                ExecutionEngine,
                typename mss_components_is_reduction< mss_components_t >::type,
                ReductionData,
                nocolor > run_functor_args_t;

            typedef boost::mpl::range_c< uint_t, 0, boost::mpl::size< functors_list_t >::type::value > iter_range;

            // now the corresponding backend has to execute all the functors of the mss
            backend_traits_from_id< BackendIds::s_backend_id >::template mss_loop< run_functor_args_t >::template run(
                local_domain, m_grid, m_reduction_data, m_block_idx, m_block_idy);
        }
    };
} // namespace gridtools

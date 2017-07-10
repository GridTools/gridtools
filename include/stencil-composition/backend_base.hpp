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

#include <boost/mpl/filter_view.hpp>
#include <boost/mpl/reverse.hpp>
#include <boost/mpl/transform.hpp>

#include "./backend_traits_fwd.hpp"
#include "./run_functor_arguments.hpp"
#include "gridtools.hpp"

#ifdef __CUDACC__
#include "./backend_cuda/backend_cuda.hpp"
#else
#include "./backend_host/backend_host.hpp"
#endif

#include "../common/meta_array.hpp"
#include "../common/pair.hpp"
#include "./accessor.hpp"
#include "./aggregator_type.hpp"
#include "./conditionals/condition.hpp"
#include "./intermediate_impl.hpp"
#include "./mss.hpp"
#include "./mss_local_domain.hpp"
#include "./mss_metafunctions.hpp"
#include "./storage_wrapper.hpp"
#include "./tile.hpp"
#include "../storage/storage-facility.hpp"
#include "./conditionals/condition.hpp"
#include "../common/stencil_serializer.hpp"

/**
   @file
   @brief base class for all the backends. Current supported backend are \ref gridtools::enumtype::Host and \ref
   gridtools::enumtype::Cuda
   It is templated on the derived type (CRTP pattern) in order to use static polymorphism.
*/

namespace gridtools {

    /**
        this struct contains the 'run' method for all backends, with a
        policy determining the specific type. Each backend contains a
        traits class for the specific case.

        backend<type, strategy>
        there are traits: one for type and one for strategy.
        - type refers to the architecture specific, like the
          differences between cuda and the host.

        The backend has a member function "run" that is called by the
        "intermediate".
        The "run" method calls strategy_from_id<strategy>::loop

        - the strategy_from_id is in the specific backend_? folder, such as
        - in backend_?/backend_traits.h

        - strategy_from_id contains the tile size information and the
        - "struct loop" which has the "run_loop" member function.

        Before calling the loop::run_loop method, the backend queries
        "execute_traits" that are contained in the
        "backend_traits_t". the latter is obtained by

        backend_strategy_from_id<type>

        The execute_traits::backend_t (bad name) is responsible for
        the "inner loop nests". The
        loop<execute_traits::backend_t>::run_loop will use that to do
        whatever he has to do, for instance, the host_backend will
        iterate over the functors of the MSS using the for_each
        available there.

        - Similarly, the definition (specialization) is contained in the
        - specific subfoled (right now in backend_?/backend_traits_?.h ).

        - This contains:
        - - (INTERFACE) pointer<>::type that returns the first argument to instantiate the storage class
        - - (INTERFACE) storage_traits::storage_t to get the storage type to be used with the backend
        - - (INTERFACE) execute_traits ?????? this was needed when backend_traits was forcely shared between host and
       cuda backends. Now they are separated and this may be simplified.
        - - (INTERNAL) for_each that is used to invoke the different things for different stencils in the MSS
        - - (INTERNAL) once_per_block
    */
    template < enumtype::platform BackendId, enumtype::grid_type GridId, enumtype::strategy StrategyId >
    struct backend_base {

#ifdef __CUDACC__
        GRIDTOOLS_STATIC_ASSERT(BackendId == enumtype::Cuda,
            "Beware: you are compiling with nvcc, and most probably "
            "want to use the cuda backend, but the backend you are "
            "instantiating is another one!!");
#endif

        typedef backend_base< BackendId, GridId, StrategyId > this_type;

        typedef backend_ids< BackendId, GridId, StrategyId > backend_ids_t;

        typedef backend_traits_from_id< BackendId > backend_traits_t;
        typedef grid_traits_from_id< GridId > grid_traits_t;
        typedef storage_traits< BackendId > storage_traits_t;
        typedef typename backend_traits_t::template select_strategy< backend_ids_t >::type strategy_traits_t;

        static const enumtype::strategy s_strategy_id = StrategyId;
        static const enumtype::platform s_backend_id = BackendId;
        static const enumtype::grid_type s_grid_type_id = GridId;

        /** types of the functions used to compute the thread grid information
            for allocating the temporary storages and such
        */
        typedef uint_t (*query_i_threads_f)(uint_t);
        typedef uint_t (*query_j_threads_f)(uint_t);

        /**
            Method to retrieve a global parameter
         */
        template < typename T >
        static typename storage_traits_t::template data_store_t< T,
            typename storage_traits_t::template special_storage_info_t< 0, selector< 0u > > >
        make_global_parameter(T const &t) {
            typename storage_traits_t::template special_storage_info_t< 0, selector< 0u > > si(1);
            typename storage_traits_t::template data_store_t< T, decltype(si) > ds(si);
            make_host_view(ds)(0) = t;
            return ds;
        }

        /**
            Method to update a global parameter
         */
        template < typename T, typename V >
        static void update_global_parameter(T &gp, V const &new_val) {
            gp.sync();
            auto view = make_host_view(gp);
            assert(check_consistency(gp, view) && "Cannot create a valid view to a global parameter. Properly synced?");
            view(0) = new_val;
            gp.sync();
        }

        /**
            Method to instantiate the views (according to the given backend)
         */
        template < typename AggregatorType, typename ViewFusionMap >
        static void instantiate_views(AggregatorType &aggregator, ViewFusionMap &viewmap) {
            GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< AggregatorType >::value), GT_INTERNAL_ERROR);
            boost::fusion::for_each(
                viewmap, typename backend_traits_t::template instantiate_view< AggregatorType >(aggregator));
        }

        /**
            Method to extract a storage_info pointer from a metadata_set
         */
        template < typename StorageInfoPtr, typename AggregatorType >
        static typename StorageInfoPtr::value_type *extract_storage_info_ptrs(AggregatorType const &aggregator) {
            GRIDTOOLS_STATIC_ASSERT(
                (is_storage_info< typename boost::decay< typename StorageInfoPtr::value_type >::type >::value),
                GT_INTERNAL_ERROR);
            return backend_traits_t::template extract_storage_info_ptr(
                aggregator.metadata_set_view().template get< StorageInfoPtr >().get());
        }

        /**
            Method to extract get a storage_info for a temporary storage (could either be a icosahedral or a standard
           storage info)
         */
        template < typename MaxExtent, typename StorageWrapper, typename Grid >
        static typename StorageWrapper::storage_info_t instantiate_storage_info(Grid const &grid) {
            GRIDTOOLS_STATIC_ASSERT(
                (is_storage_info< typename StorageWrapper::storage_info_t >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_storage_wrapper< StorageWrapper >::value), GT_INTERNAL_ERROR);
            return grid_traits_t::template instantiate_storage_info< MaxExtent, this_type, StorageWrapper >(grid);
        }

        /**
         * @brief metafunction that computes the map of all the temporaries and their associated ij extents
         * @tparam AggregatorType domain type containing the placeholders for all storages (including temporaries)
         * @tparam MssComponents the mss components of the MSS
         * @output map of <temporary placeholder, extent> where the extent is the enclosing extent of all the extents
         *      defined for the different functors of a MSS.
         */
        template < typename AggregatorType, typename MssComponents >
        struct obtain_map_extents_temporaries_mss {
            GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< AggregatorType >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_mss_components< MssComponents >::value), GT_INTERNAL_ERROR);
            typedef typename MssComponents::extent_sizes_t ExtendSizes;

            // filter all the temporary args
            typedef typename boost::mpl::fold< typename AggregatorType::placeholders_t,
                boost::mpl::vector0<>,
                boost::mpl::if_< is_tmp_arg< boost::mpl::_2 >,
                                                   boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 >,
                                                   boost::mpl::_1 > >::type list_of_temporaries;

            // vector of written temporaries per functor (vector of vectors)
            typedef typename MssComponents::written_temps_per_functor_t written_temps_per_functor_t;

            typedef typename boost::mpl::fold< list_of_temporaries,
                boost::mpl::map0<>,
                _impl::associate_extents_map< boost::mpl::_1,
                                                   boost::mpl::_2,
                                                   written_temps_per_functor_t,
                                                   ExtendSizes > >::type type;
        };

        /**
         * @brief metafunction that merges two maps of <temporary, ij extent>
         * The merge is performed by computing the union of all the extents found associated
         * to the same temporary, i.e. the enclosing extent.
         * @tparam extent_map1 first map to merge
         * @tparam extent_map2 second map to merge
          */
        template < typename extent_map1, typename extent_map2 >
        struct merge_extent_temporary_maps {
            typedef typename boost::mpl::fold<
                extent_map1,
                extent_map2,
                boost::mpl::if_< boost::mpl::has_key< extent_map2, boost::mpl::first< boost::mpl::_2 > >,
                    boost::mpl::insert< boost::mpl::_1,
                                     boost::mpl::pair< boost::mpl::first< boost::mpl::_2 >,
                                            enclosing_extent< boost::mpl::second< boost::mpl::_2 >,
                                                           boost::mpl::at< extent_map2,
                                                                  boost::mpl::first< boost::mpl::_2 > > > > >,
                    boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > > >::type type;
        };

        /**
         * @brief metafunction that computes the map of all the temporaries and their associated ij extents
         * for all the Mss components in an array (corresponding to a Computation)
         * @tparam AggregatorType domain type containing the placeholders for all storages (including temporaries)
         * @tparam MssComponentsArray meta array of the mss components of all MSSs
         * @output map of <temporary placeholder, extent> where the extent is the enclosing extent of all the extents
         *      defined for the temporary in all MSSs.
         */
        template < typename AggregatorType, typename MssComponentsArray >
        struct obtain_map_extents_temporaries_mss_array {
            GRIDTOOLS_STATIC_ASSERT(
                (is_meta_array_of< MssComponentsArray, is_mss_components >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< AggregatorType >::value), GT_INTERNAL_ERROR);

            typedef typename boost::mpl::fold<
                typename MssComponentsArray::elements,
                boost::mpl::map0<>,
                merge_extent_temporary_maps< boost::mpl::_1,
                    obtain_map_extents_temporaries_mss< AggregatorType, boost::mpl::_2 > > >::type type;
        };

        template < typename AggregatorType, typename MssArray1, typename MssArray2, typename Cond >
        struct obtain_map_extents_temporaries_mss_array< AggregatorType, condition< MssArray1, MssArray2, Cond > > {
            GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< AggregatorType >::value), GT_INTERNAL_ERROR);

            typedef typename obtain_map_extents_temporaries_mss_array< AggregatorType, MssArray1 >::type type1;
            typedef typename obtain_map_extents_temporaries_mss_array< AggregatorType, MssArray2 >::type type2;
            typedef
                typename boost::mpl::fold< type2, type1, boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > >::type
                    type;
        };

        /**
         * @brief compute a list with all the storage_wrappers
         * @tparam AggregatorType domain
         * @tparam MssComponentsArray meta array of mss components
         */
        template < typename AggregatorType, typename MssComponentsArray >
        struct obtain_storage_wrapper_list_t {

            GRIDTOOLS_STATIC_ASSERT((is_condition< MssComponentsArray >::value ||
                                        is_meta_array_of< MssComponentsArray, is_mss_components >::value),
                GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< AggregatorType >::value), GT_INTERNAL_ERROR);

            typedef typename backend_traits_t::template get_block_size< StrategyId >::type block_size_t;

            static const uint_t tileI = block_size_t::i_size_t::value;
            static const uint_t tileJ = block_size_t::j_size_t::value;

            typedef typename obtain_map_extents_temporaries_mss_array< AggregatorType, MssComponentsArray >::type
                map_of_extents;

            typedef typename boost::mpl::fold<
                map_of_extents,
                boost::mpl::vector0<>,
                boost::mpl::push_back< boost::mpl::_1,
                    typename _impl::get_storage_wrapper< tileI, tileJ >::template apply< boost::mpl::_2 > > >::type
                type;
        };

        /**
         * \brief calls the \ref gridtools::run_functor for each functor in the FunctorList.
         * the loop over the functors list is unrolled at compile-time using the for_each construct.
         * @tparam MssArray  meta array of mss
         * \tparam Domain Domain class (not really useful maybe)
         * \tparam Grid Coordinate class with domain sizes and splitter grid
         * \tparam MssLocalDomainArray sequence of mss local domain (containing each the sequence of local domain list)
         */
        template < typename MssComponentsArray,
            typename Grid,
            typename MssLocalDomainArray,
            typename ReductionData > // List of local domain to be pbassed to functor at<i>
        static void
        run(/*Domain const& domain, */ Grid const &grid,
            MssLocalDomainArray &mss_local_domain_list,
            ReductionData &reduction_data) {
            // TODO: I would swap the arguments coords and local_domain_list here, for consistency
            GRIDTOOLS_STATIC_ASSERT(
                (is_sequence_of< MssLocalDomainArray, is_mss_local_domain >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT(
                (is_meta_array_of< MssComponentsArray, is_mss_components >::value), GT_INTERNAL_ERROR);

            strategy_traits_t::template fused_mss_loop< MssComponentsArray, backend_ids_t, ReductionData >::run(
                mss_local_domain_list, grid, reduction_data);
        }

        /**
         * \brief Calls the \ref gridtools::run_functor for each functor in the FunctorList.
         *
         * \see
         *    gridtools::backend_base::run
         */
        template < typename MssComponentsArray,
            typename Grid,
            typename MssLocalDomainArray,
            typename ReductionData,
            typename SerializerType >
        static void run_and_serialize(Grid const &grid,
            MssLocalDomainArray &mss_local_domain_list,
            ReductionData &reduction_data,
            stencil_serializer< SerializerType > &stencil_ser) {

            GRIDTOOLS_STATIC_ASSERT(
                (is_sequence_of< MssLocalDomainArray, is_mss_local_domain >::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT(
                (is_meta_array_of< MssComponentsArray, is_mss_components >::value), "Internal Error: wrong type");

            strategy_traits_t::template fused_mss_loop< MssComponentsArray,
                backend_ids_t,
                ReductionData >::run_and_serialize(mss_local_domain_list, grid, reduction_data, stencil_ser);
        }

        /** Initial interface

            Threads are oganized in a 2D grid. These two functions
            n_i_pes() and n_j_pes() retrieve the
            information about how to compute those sizes.

            The information needed by those functions are the sizes of the
            domains (especially if the GPU is used)

            n_i_pes()(size): number of threads on the first dimension of the thread grid
        */
        static query_i_threads_f n_i_pes() { return &backend_traits_t::n_i_pes; }

        /** Initial interface

            Threads are oganized in a 2D grid. These two functions
            n_i_pes() and n_j_pes() retrieve the
            information about how to compute those sizes.

            The information needed by those functions are the sizes of the
            domains (especially if the GPU is used)

            n_j_pes()(size): number of threads on the second dimension of the thread grid
        */
        static query_j_threads_f n_j_pes() { return &backend_traits_t::n_j_pes; }
    };

} // namespace gridtools

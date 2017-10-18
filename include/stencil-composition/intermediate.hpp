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

#ifdef VERBOSE
#include <iostream>
#endif
#include <utility>

#include <boost/shared_ptr.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/copy.hpp>
#include <boost/fusion/include/copy.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/transform.hpp>
#include <boost/fusion/include/any.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/list.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/mpl/min_element.hpp>
#include <boost/mpl/max_element.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/copy.hpp>
#include <boost/type_traits/remove_const.hpp>
#include "../common/meta_array.hpp"
#include "./amss_descriptor.hpp"
#include "./backend_base.hpp"
#include "./backend_metafunctions.hpp"
#include "./backend_traits_fwd.hpp"
#include "./computation.hpp"
#include "./compute_extents_metafunctions.hpp"
#include "./conditionals/switch_variable.hpp"
#include "./esf.hpp"
#include "./functor_do_method_lookup_maps.hpp"
#include "./functor_do_methods.hpp"
#include "./grid.hpp"
#include "./grid_traits.hpp"
#include "./intermediate_impl.hpp"
#include "./level.hpp"
#include "./local_domain.hpp"
#include "./loopintervals.hpp"
#include "./mss_components_metafunctions.hpp"
#include "./mss_local_domain.hpp"
#include "./reductions/reduction_data.hpp"
#include "./storage_wrapper.hpp"
#include "./wrap_type.hpp"
#include "iterate_on_esfs.hpp"

/**
 * @file
 * \brief this file contains mainly helper metafunctions which simplify the interface for the application developer
 * */
namespace gridtools {

    template < typename T >
    struct if_condition_extract_index_t;

    template < enumtype::platform >
    struct setup_computation;

    template <>
    struct setup_computation< enumtype::Cuda > {

        template < typename AggregatorType, typename Grid >
        static uint_t apply(AggregatorType &aggregator, Grid const &grid) {
            GRIDTOOLS_STATIC_ASSERT(
                is_aggregator_type< AggregatorType >::value, GT_INTERNAL_ERROR_MSG("wrong domain type"));
            GRIDTOOLS_STATIC_ASSERT(is_grid< Grid >::value, GT_INTERNAL_ERROR_MSG("wrong grid type"));
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of< typename AggregatorType::arg_storage_pair_fusion_list_t,
                                        is_arg_storage_pair >::type::value),
                GT_INTERNAL_ERROR_MSG("wrong type: the aggregator_type contains non arg_storage_pairs in "
                                      "arg_storage_pair_fusion_list_t"));
            grid.clone_to_device();
            return GT_NO_ERRORS;
        }
    };

    template <>
    struct setup_computation< enumtype::Host > {
        template < typename AggregatorType, typename Grid >
        static uint_t apply(AggregatorType &aggregator, Grid const &grid) {
            GRIDTOOLS_STATIC_ASSERT(
                is_aggregator_type< AggregatorType >::value, GT_INTERNAL_ERROR_MSG("wrong domain type"));
            GRIDTOOLS_STATIC_ASSERT(is_grid< Grid >::value, GT_INTERNAL_ERROR_MSG("wrong grid type"));
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of< typename AggregatorType::arg_storage_pair_fusion_list_t,
                                        is_arg_storage_pair >::type::value),
                GT_INTERNAL_ERROR_MSG("wrong type: the aggregator_type contains non arg_storage_pairs in "
                                      "arg_storage_pair_fusion_list_t"));

            return GT_NO_ERRORS;
        }
    };

    /**
     * @brief metafunction that create the mss local domain type
     */
    template < enumtype::platform BackendId, typename MssComponentsArray, typename StorageWrapperList, bool IsStateful >
    struct create_mss_local_domains {

        GRIDTOOLS_STATIC_ASSERT((is_meta_array_of< MssComponentsArray, is_mss_components >::value), GT_INTERNAL_ERROR);

        struct get_the_mss_local_domain {
            template < typename T >
            struct apply {
                typedef mss_local_domain< BackendId, T, StorageWrapperList, IsStateful > type;
            };
        };

        typedef typename boost::mpl::transform< typename MssComponentsArray::elements, get_the_mss_local_domain >::type
            type;
    };

    template < enumtype::platform BackendId,
        typename MssArray1,
        typename MssArray2,
        typename Cond,
        typename StorageWrapperList,
        bool IsStateful >
    struct create_mss_local_domains< BackendId,
        condition< MssArray1, MssArray2, Cond >,
        StorageWrapperList,
        IsStateful > {
        typedef typename create_mss_local_domains< BackendId, MssArray1, StorageWrapperList, IsStateful >::type type1;
        typedef typename create_mss_local_domains< BackendId, MssArray2, StorageWrapperList, IsStateful >::type type2;
        typedef condition< type1, type2, Cond > type;
    };

    template < typename AggregatorType >
    struct create_view_fusion_map {
        GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< AggregatorType >::value),
            GT_INTERNAL_ERROR_MSG("Given type is not an aggregator_type."));

        // get all the storages from the placeholders
        typedef typename boost::mpl::fold< typename AggregatorType::placeholders_t,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1, get_storage_from_arg< boost::mpl::_2 > > >::type storage_list_t;
        // convert the storages into views
        typedef typename boost::mpl::transform< storage_list_t, _impl::get_view_t >::type data_views_t;
        // equip with args
        typedef typename boost::mpl::fold<
            boost::mpl::range_c< unsigned, 0, AggregatorType::len >,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1,
                boost::fusion::pair< boost::mpl::at< typename AggregatorType::placeholders_t, boost::mpl::_2 >,
                                       boost::mpl::at< data_views_t, boost::mpl::_2 > > > >::type arg_to_view_vec;
        // fusion map from args to views
        typedef typename boost::fusion::result_of::as_map< arg_to_view_vec >::type type;
    };

    template < typename Backend, typename AggregatorType, typename ViewList, typename MssComponentsArray >
    struct create_storage_wrapper_list {
        // handle all tmps, obtain the storage_wrapper_list for written tmps
        typedef typename Backend::template obtain_storage_wrapper_list_t< AggregatorType, MssComponentsArray >::type
            all_tmps;

        // for every placeholder we push back an element that is either a new storage_wrapper type
        // for a normal data_store(_field), or in case it is a tmp we get the element out of the all_tmps list.
        // if we find a read-only tmp void will be pushed back, but this will be filtered out in the
        // last step.
        typedef boost::mpl::range_c< int, 0, AggregatorType::len > iter_range;
        typedef typename boost::mpl::fold<
            iter_range,
            boost::mpl::vector0<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                boost::mpl::if_<
                    is_tmp_arg< boost::mpl::at< typename AggregatorType::placeholders_t, boost::mpl::_2 > >,
                    storage_wrapper_elem< boost::mpl::at< typename AggregatorType::placeholders_t, boost::mpl::_2 >,
                        all_tmps >,
                    storage_wrapper< boost::mpl::at< typename AggregatorType::placeholders_t, boost::mpl::_2 >,
                        boost::mpl::at< ViewList, boost::mpl::_2 >,
                        tile< 0, 0, 0 >,
                        tile< 0, 0, 0 > > > > >::type complete_list;
        // filter the list
        typedef
            typename boost::mpl::filter_view< complete_list, is_storage_wrapper< boost::mpl::_1 > >::type filtered_list;
        typedef typename boost::mpl::fold< filtered_list,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 > >::type type;
    };

    template < typename IsPresent, typename MssComponentsArray, typename Backend >
    struct run_conditionally;

    /**@brief calls the run method when conditionals are defined

       specialization for when the next MSS is not a conditional
    */
    template < typename MssComponentsArray, typename Backend >
    struct run_conditionally< boost::mpl::true_, MssComponentsArray, Backend > {
        template < typename ConditionalSet, typename Grid, typename MssLocalDomainList, typename ReductionData >
        static void apply(ConditionalSet const & /**/,
            Grid const &grid_,
            MssLocalDomainList const &mss_local_domain_list_,
            ReductionData &reduction_data) {
            Backend::template run< MssComponentsArray >(grid_, mss_local_domain_list_, reduction_data);
        }
    };

    /**
       @brief calls the run method when conditionals are defined

       specialization for when the next MSS is a conditional
     */
    template < typename Array1, typename Array2, typename Cond, typename Backend >
    struct run_conditionally< boost::mpl::true_, condition< Array1, Array2, Cond >, Backend > {
        template < typename ConditionalSet, typename Grid, typename MssLocalDomainList, typename ReductionData >
        static void apply(ConditionalSet const &conditionals_set_,
            Grid const &grid_,
            MssLocalDomainList const &mss_local_domain_list_,
            ReductionData &reduction_data) {
            // std::cout<<"true? "<<boost::fusion::at_key< Cond >(conditionals_set_).value()<<std::endl;
            if (boost::fusion::at_key< Cond >(conditionals_set_).value()) {
                run_conditionally< boost::mpl::true_, Array1, Backend >::apply(
                    conditionals_set_, grid_, mss_local_domain_list_, reduction_data);
            } else
                run_conditionally< boost::mpl::true_, Array2, Backend >::apply(
                    conditionals_set_, grid_, mss_local_domain_list_, reduction_data);
        }
    };

    /**@brief calls the run method when no conditional is defined

       the 2 cases are separated into 2 different partial template specialization, because
       the fusion::at_key doesn't compile when the key is not present in the set
       (i.e. the present situation).
     */
    template < typename MssComponentsArray, typename Backend >
    struct run_conditionally< boost::mpl::false_, MssComponentsArray, Backend > {
        template < typename ConditionalSet, typename Grid, typename MssLocalDomainList, typename ReductionData >
        static void apply(ConditionalSet const &,
            Grid const &grid_,
            MssLocalDomainList const &mss_local_domain_list_,
            ReductionData &reduction_data) {

            Backend::template run< MssComponentsArray >(grid_, mss_local_domain_list_, reduction_data);
        }
    };

    template < typename Vec >
    struct extract_mss_domains {
        typedef Vec type;
    };

    template < typename Vec1, typename Vec2, typename Cond >
    struct extract_mss_domains< condition< Vec1, Vec2, Cond > > {

        // TODO: how to do the check described below?
        // GRIDTOOLS_STATIC_ASSERT((boost::is_same<typename extract_mss_domains<Vec1>::type, typename
        // extract_mss_domains<Vec2>::type>::type::value), "The case in which 2 different mss are enabled/disabled using
        // conditionals is supported only when they work with the same placeholders. Here you are trying to switch
        // between MSS for which the type (or the order) of the placeholders is not the same");
        // consider the first one
        typedef typename extract_mss_domains< Vec1 >::type type;
    };

    template < typename MssDescriptorSequence >
    struct need_to_compute_extents {

        /* helper since boost::mpl::and_ fails in this case with nvcc
        */
        template < typename BoolA, typename BoolB >
        struct gt_and {
            using type = typename boost::mpl::bool_< BoolA::value and BoolB::value >;
        };

        /* helper since boost::mpl::or_ fails in this case with nvcc
        */
        template < typename BoolA, typename BoolB >
        struct gt_or {
            using type = typename boost::mpl::bool_< BoolA::value or BoolB::value >;
        };

        using has_all_extents = typename with_operators< is_esf_with_extent,
            gt_and >::template iterate_on_esfs< boost::mpl::bool_< true >, MssDescriptorSequence >::type;
        using has_extent = typename with_operators< is_esf_with_extent,
            gt_or >::template iterate_on_esfs< boost::mpl::bool_< false >, MssDescriptorSequence >::type;

        GRIDTOOLS_STATIC_ASSERT((has_extent::value == has_all_extents::value),
            "The computation appears to have stages with and without extents being specified at the same time. A "
            "computation should have all stages with extents or none.");
        using type = typename boost::mpl::not_< has_all_extents >::type;
    };

    template < bool do_compute_extents,
        typename MssElements,
        typename GridTraits,
        typename Placeholders,
        uint_t RepeatFunctor >
    struct obtain_extents_to_esfs_map {
        // First we need to compute the association between placeholders and extents.
        // This information is needed to allocate temporaries, and to provide the
        // extent information to the user.
        typedef typename placeholder_to_extent_map< MssElements, GridTraits, Placeholders, RepeatFunctor >::type
            extent_map_t;

        // Second we need to associate an extent to each esf, so that
        // we can associate loop bounds to the functors.
        typedef typename associate_extents_to_esfs< MssElements, extent_map_t, RepeatFunctor >::type type;
    };

    template < typename MssElements, typename GridTraits, typename Placeholders, uint_t RepeatFunctor >
    struct obtain_extents_to_esfs_map< false, MssElements, GridTraits, Placeholders, RepeatFunctor > {

        /** helper function _ there is a specialization for
            independent stages.
        */
        template < typename CurrentList, typename EsfElem >
        struct enqueue_extents {
            using type = typename boost::mpl::push_back< CurrentList, typename esf_extent< EsfElem >::type >::type;
        };

        /** Specialization for independent stages
         */
        template < typename CurrentList, typename... EsfElems >
        struct enqueue_extents< CurrentList, independent_esf< EsfElems... > > {
            using list = typename independent_esf< EsfElems... >::esf_list;
            using type = typename boost::mpl::fold< list,
                CurrentList,
                boost::mpl::push_back< boost::mpl::_1, esf_extent< boost::mpl::_2 > > >::type;
        };

        template < typename MssDescriptor >
        struct get_esf_extents {
            using type = typename boost::mpl::fold< typename MssDescriptor::esf_sequence_t,
                boost::mpl::vector0<>,
                enqueue_extents< boost::mpl::_1, boost::mpl::_2 > >::type;
        };

        using type = typename boost::mpl::fold< MssElements,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1, get_esf_extents< boost::mpl::_2 > > >::type;
    };

    // function that checks if the given extents (I+- and J+-)
    // are within the halo that was defined when creating the grid.
    template < typename ExtentsVec, typename Grid >
    void check_grid_against_extents(Grid const &grid) {
        typedef ExtentsVec all_extents_vecs_t;
        // get smallest i_minus extent
        typedef typename boost::mpl::deref<
            typename boost::mpl::min_element< typename boost::mpl::transform< all_extents_vecs_t,
                boost::mpl::lambda< boost::mpl::at< boost::mpl::_1, boost::mpl::int_< 0 > > >::type >::type >::type >::
            type IM_t;
        // get smallest j_minus extent
        typedef typename boost::mpl::deref<
            typename boost::mpl::min_element< typename boost::mpl::transform< all_extents_vecs_t,
                boost::mpl::lambda< boost::mpl::at< boost::mpl::_1, boost::mpl::int_< 2 > > >::type >::type >::type >::
            type JM_t;
        // get largest i_plus extent
        typedef typename boost::mpl::deref<
            typename boost::mpl::max_element< typename boost::mpl::transform< all_extents_vecs_t,
                boost::mpl::lambda< boost::mpl::at< boost::mpl::_1, boost::mpl::int_< 1 > > >::type >::type >::type >::
            type IP_t;
        // get largest j_plus extent
        typedef typename boost::mpl::deref<
            typename boost::mpl::max_element< typename boost::mpl::transform< all_extents_vecs_t,
                boost::mpl::lambda< boost::mpl::at< boost::mpl::_1, boost::mpl::int_< 3 > > >::type >::type >::type >::
            type JP_t;
        const bool check = (IM_t::value >= -static_cast< int >(grid.direction_i().minus())) &&
                           (IP_t::value <= static_cast< int >(grid.direction_i().plus())) &&
                           (JM_t::value >= -static_cast< int >(grid.direction_j().minus())) &&
                           (JP_t::value <= static_cast< int >(grid.direction_j().plus()));
        assert(check && "One of the stencil accessor extents is exceeding the halo region.");
    }

    namespace _impl {
        /**
           This is a functor used to iterate with boost::fusion::any
           to check that grid size is small enough to not make the
           stencil go out of bound on data fields.

           \tparam GridTraits The grid traits of the grid in question to get the indices of relevant coordinates
           \tparam Grid The Grid
        */
        template < typename GridTraits, typename Grid >
        struct check_with {
            Grid const &grid;

            check_with(Grid const &grid) : grid(grid) {}

            /**
               The element of the metadata set that describe the sizes
               of the storages. boost::fusion::any is stopping
               iteration when a `true` is returned, so the iteration
               returns `false` when the check passes.

               \tparam The type element of a metadata_set which is a gridtools::pointer to a metadata
               \param mde The element of a metadata_set which is a gridtools::pointer to a metadata
             */
            template < typename MetaDataElem >
            bool operator()(MetaDataElem const &mde) const {
                bool result = true;

                // Here we need to use the at_ interface instead of
                // the at, since at_ does not assert out-of-bound
                // queries, but actually returns -1.

                // TODO: This check may be not accurate since there is
                // an ongoing change in the convention for storage and
                // grid. Before the storage had the conventions that
                // there was not distinction between halo and core
                // region in the storage. The distinction was made
                // solely in the grid. Now the storage makes that
                // distinction, ad when aqllocating the data the halo
                // is also allocated. So for instance a stoage of
                // 3x3x3 with halo of <1,1,1> will allocate a 5x5x5
                // storage. The grid is the same as before. The first
                // step will be to update the storage to point as
                // first eleent the (1,1,1) element and then to get
                // the grid to not specifying halos (at least in the
                // simple cases). This is why the check is left as
                // before here, but may be updated with more accurate
                // ones when the convention is updated
                if (MetaDataElem::value_type::layout_t::template at_< GridTraits::dim_k_t::value >::value >= 0) {
                    result = result && (grid.k_max() + 1 <= mde->template dim< GridTraits::dim_k_t::value >());
                }

                if (MetaDataElem::value_type::layout_t::template at_< GridTraits::dim_j_t::value >::value >= 0) {
                    result = result && (grid.j_high_bound() + 1 <= mde->template dim< GridTraits::dim_j_t::value >());
                }

                if (MetaDataElem::value_type::layout_t::template at_< GridTraits::dim_i_t::value >::value >= 0) {
                    result = result && (grid.i_high_bound() + 1 <= mde->template dim< GridTraits::dim_i_t::value >());
                }

                return !result;
            }
        };
    } // namespace _impl

    /**
       Given the Aggregator this function checks that the
       iteration space of the grid would not cause out of bound
       accesses from the stencil execution. This function is
       automatically called when constructing a computation.

       \tparam GridTraits The traits in the grid in question to get the indices of the relevant coordinates
       \tparam Grid The type of the grid (normally deduced by the argument)
       \tparam Aggregator The aggregator (normally deduced by the argument)

       \param grid The grid to check
       \param aggrs The aggregator
    */
    template < typename GridTraits, typename Grid, typename Aggregator >
    void check_fields_sizes(Grid const &grid, Aggregator const &aggr) {
        auto metadata_view = aggr.metadata_set_view().sequence_view();
        bool is_wrong = boost::fusion::any(metadata_view, _impl::check_with< GridTraits, Grid >(grid));
        if (is_wrong) {
            throw std::runtime_error(
                "Error: Iteration space size is bigger than some storages sizes, this would likely "
                "result in access violation. Please check storage sizes against grid sizes, "
                "including the axis levels.");
        }
    }

    /**
     * @class
     *  @brief structure collecting helper metafunctions
     */
    template < typename Backend,
        typename MssDescriptorArrayIn,
        typename DomainType,
        typename Grid,
        typename ConditionalsSet,
        typename ReductionType,
        bool IsStateful,
        uint_t RepeatFunctor >
    struct intermediate : public computation< DomainType, ReductionType > {
        // fix the temporaries by replacing the given storage info index with a new one
        // fix the and expandable parameters by replacing the vector type with an expandable_paramter type

        using base_t = computation< DomainType, ReductionType >;

        typedef
            typename fix_mss_arg_indices< MssDescriptorArrayIn, DomainType, RepeatFunctor >::type MssDescriptorArray;

        GRIDTOOLS_STATIC_ASSERT(
            (is_meta_array_of< MssDescriptorArray, is_computation_token >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_backend< Backend >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< DomainType >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);
        // GRIDTOOLS_STATIC_ASSERT((is_conditionals_set<ConditionalsSet>::value), GT_INTERNAL_ERROR);

        typedef ConditionalsSet conditionals_set_t;
        typedef typename Backend::backend_traits_t::performance_meter_t performance_meter_t;
        typedef typename Backend::backend_ids_t backend_ids_t;
        typedef grid_traits_from_id< backend_ids_t::s_grid_type_id > grid_traits_t;

        // substituting the std::vector type in the args<> with a correspondent
        // expandable_parameter placeholder
        typedef typename substitute_expandable_params< typename DomainType::placeholders_t, RepeatFunctor >::type
            placeholders_t;

        typedef typename obtain_extents_to_esfs_map<
            need_to_compute_extents< typename MssDescriptorArray::elements >::type::value,
            typename MssDescriptorArray::elements,
            grid_traits_t,
            placeholders_t,
            RepeatFunctor >::type extent_sizes_t;

        typedef typename boost::mpl::if_<
            boost::mpl::is_sequence< typename MssDescriptorArray::elements >,
            typename boost::mpl::fold< typename MssDescriptorArray::elements,
                boost::mpl::false_,
                boost::mpl::or_< boost::mpl::_1, mss_descriptor_is_reduction< boost::mpl::_2 > > >::type,
            boost::mpl::false_ >::type has_reduction_t;

        typedef reduction_data< MssDescriptorArray, has_reduction_t::value > reduction_data_t;
        typedef typename reduction_data_t::reduction_type_t reduction_type_t;
        GRIDTOOLS_STATIC_ASSERT((boost::is_same< reduction_type_t, ReductionType >::value),
            "Error deducing the reduction. Check that if there is a reduction, this appears in the last mss");

        typedef typename build_mss_components_array< backend_id< Backend >::value,
            MssDescriptorArray,
            extent_sizes_t,
            static_int< RepeatFunctor >,
            typename Grid::axis_type >::type mss_components_array_t;

        // creates a fusion sequence of views
        typedef typename create_view_fusion_map< DomainType >::type view_list_fusion_t;

        // create storage_wrapper_list
        typedef typename create_storage_wrapper_list< Backend,
            DomainType,
            typename create_view_fusion_map< DomainType >::data_views_t,
            mss_components_array_t >::type storage_wrapper_list_t;

        // create storage_wrapper_fusion_list
        typedef
            typename boost::fusion::result_of::as_vector< storage_wrapper_list_t >::type storage_wrapper_fusion_list_t;

        // get the maximum extent (used to retrieve the size of the temporaries)
        typedef typename max_i_extent_from_storage_wrapper_list< storage_wrapper_fusion_list_t >::type max_i_extent_t;

        // creates an mpl sequence of local domains
        typedef typename create_mss_local_domains< backend_id< Backend >::value,
            mss_components_array_t,
            storage_wrapper_list_t,
            IsStateful >::type mss_local_domains_t;

        // creates a fusion vector of local domains
        typedef typename boost::fusion::result_of::as_vector<
            typename extract_mss_domains< mss_local_domains_t >::type >::type mss_local_domain_list_t;

        // member fields
        mss_local_domain_list_t m_mss_local_domain_list;

        const Grid m_grid;

        bool is_storage_ready;
        performance_meter_t m_meter;

        conditionals_set_t m_conditionals_set;
        reduction_data_t m_reduction_data;
        view_list_fusion_t m_view_list;
        storage_wrapper_fusion_list_t m_storage_wrapper_list;

        using base_t::m_domain;

      public:
        intermediate(DomainType const &domain,
            Grid const &grid,
            ConditionalsSet conditionals_,
            typename reduction_data_t::reduction_type_t reduction_initial_value = 0)
            : base_t(domain), m_grid(grid), m_meter("NoName"), m_conditionals_set(conditionals_),
              m_reduction_data(reduction_initial_value) {
            // check_grid_against_extents< all_extents_vecs_t >(grid);
            // check_fields_sizes< grid_traits_t >(grid, domain);
        }
        /**
           @brief This method allocates on the heap the temporary variables.
           Calls heap_allocated_temps::prepare_temporaries(...).
           It allocates the memory for the list of extents defined in the temporary placeholders.
           Further it takes care of updating the global_parameters
        */

        virtual void ready() {
            // instantiate all the temporaries
            boost::mpl::for_each< storage_wrapper_fusion_list_t >(
                _impl::instantiate_tmps< max_i_extent_t, DomainType, Grid, Backend >(m_domain, m_grid));
        }

        virtual void steady() {
            // sync the data stores that should be synced
            boost::fusion::for_each(m_domain.get_arg_storage_pairs(), _impl::sync_data_stores());
            // fill view list
            Backend::template instantiate_views< DomainType, view_list_fusion_t >(m_domain, m_view_list);
            // fill storage_wrapper_list
            boost::fusion::for_each(
                m_storage_wrapper_list, _impl::initialize_storage_wrappers< view_list_fusion_t >(m_view_list));
            // setup the computation for given backend (e.g., move grid to device)
            setup_computation< Backend::s_backend_id >::template apply(m_domain, m_grid);
            // instantiate mss_local_domains and wrapped local_domains with the right view_wrappers
            boost::fusion::for_each(m_mss_local_domain_list,
                _impl::instantiate_mss_local_domain< Backend, storage_wrapper_fusion_list_t, DomainType, IsStateful >(
                                        m_storage_wrapper_list, m_domain));
        }

        virtual void finalize() {
            // sync the data stores that should be synced
            boost::fusion::for_each(m_domain.get_arg_storage_pairs(), _impl::sync_data_stores());

            auto &all_arg_storage_pairs = m_domain.get_arg_storage_pairs();
            boost::fusion::filter_view< typename DomainType::arg_storage_pair_fusion_list_t,
                is_arg_storage_pair_to_tmp< boost::mpl::_ > > filter(all_arg_storage_pairs);
            boost::fusion::for_each(filter, _impl::delete_tmp_data_store());
        }

        virtual reduction_type_t run() {
            // check if all views are still consistent, otherwise we have to call steady again
            _impl::check_view_consistency< DomainType > check_views(m_domain);
            boost::fusion::for_each(m_view_list, check_views);
            if (!check_views.is_consistent()) {
                steady();
            }

            // typedef allowing compile-time dispatch: we separate the path when the first
            // multi stage stencil is a conditional
            typedef typename boost::fusion::result_of::has_key< conditionals_set_t,
                typename if_condition_extract_index_t< mss_components_array_t >::type >::type is_present_t;

            m_meter.start();
            run_conditionally< is_present_t, mss_components_array_t, Backend >::apply(
                m_conditionals_set, m_grid, m_mss_local_domain_list, m_reduction_data);
            m_meter.pause();
            return m_reduction_data.reduced_value();
        }

        virtual std::string print_meter() { return m_meter.to_string(); }

        virtual double get_meter() { return m_meter.total_time(); }

        virtual void reset_meter() { m_meter.reset(); }

        mss_local_domain_list_t const &mss_local_domain_list() const { return m_mss_local_domain_list; }

        void reassign_aggregator(DomainType &new_domain) { m_domain = new_domain; }
    };

} // namespace gridtools

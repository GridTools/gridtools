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

#include <utility>

#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/copy.hpp>
#include <boost/fusion/include/copy.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/transform.hpp>
#include <boost/fusion/include/any.hpp>
#include <boost/fusion/include/copy.hpp>
#include <boost/fusion/include/invoke.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/list.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/mpl/min_element.hpp>
#include <boost/mpl/max_element.hpp>

#include "../common/generic_metafunctions/copy_into_variadic.hpp"

#include "amss_descriptor.hpp"
#include "backend_base.hpp"
#include "backend_metafunctions.hpp"
#include "backend_traits_fwd.hpp"
#include "computation.hpp"
#include "compute_extents_metafunctions.hpp"
#include "conditionals/condition_tree.hpp"
#include "conditionals/switch_variable.hpp"
#include "esf.hpp"
#include "functor_do_method_lookup_maps.hpp"
#include "functor_do_methods.hpp"
#include "grid.hpp"
#include "grid_traits.hpp"
#include "intermediate_impl.hpp"
#include "level.hpp"
#include "local_domain.hpp"
#include "loopintervals.hpp"
#include "mss_components_metafunctions.hpp"
#include "mss_local_domain.hpp"
#include "reductions/reduction_data.hpp"
#include "storage_wrapper.hpp"
#include "wrap_type.hpp"
#include "make_computation_helper_cxx11.hpp"

#include "../common/meta_array_generator.hpp"
#include "computation_grammar.hpp"
#include "make_computation_helper_cxx11.hpp"
#include "all_args_in_aggregator.hpp"

/**
 * @file
 * \brief this file contains mainly helper metafunctions which simplify the interface for the application developer
 * */
namespace gridtools {

    /**
     * @brief metafunction that create the mss local domain type
     */
    template < enumtype::platform BackendId, typename MssComponents, typename StorageWrapperList, bool IsStateful >
    struct create_mss_local_domains {

        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< MssComponents, is_mss_components >::value), GT_INTERNAL_ERROR);

        struct get_the_mss_local_domain {
            template < typename T >
            struct apply {
                typedef mss_local_domain< BackendId, T, StorageWrapperList, IsStateful > type;
            };
        };

        typedef typename boost::mpl::transform< MssComponents, get_the_mss_local_domain >::type type;
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

    template < typename Placeholder >
    struct create_view {
        using type = typename _impl::get_view_t::apply< typename get_data_store_from_arg< Placeholder >::type >::type;
    };

    template < typename AggregatorType >
    struct create_view_fusion_map {
        GRIDTOOLS_STATIC_ASSERT(
            (is_aggregator_type< AggregatorType >::value), "Internal Error: Given type is not an aggregator_type.");

        using arg_and_view_seq = typename boost::mpl::transform_view< typename AggregatorType::placeholders_t,
            boost::fusion::pair< boost::mpl::_, create_view< boost::mpl::_ > > >::type;
        using type = typename boost::fusion::result_of::as_map< arg_and_view_seq >::type;
    };

    template < typename Backend, typename AggregatorType, typename MssComponentsArray >
    struct create_storage_wrapper_list {
        // handle all tmps, obtain the storage_wrapper_list for written tmps
        typedef
            typename _impl::obtain_storage_wrapper_list_t< Backend, AggregatorType, MssComponentsArray >::type all_tmps;

        // for every placeholder we push back an element that is either a new storage_wrapper type
        // for a normal data_store(_field), or in case it is a tmp we get the element out of the all_tmps list.
        // if we find a read-only tmp void will be pushed back, but this will be filtered out in the
        // last step.
        typedef typename boost::mpl::transform_view<
            typename AggregatorType::placeholders_t,
            boost::mpl::if_< is_tmp_arg< boost::mpl::_ >,
                storage_wrapper_elem< boost::mpl::_, all_tmps >,
                storage_wrapper< boost::mpl::_, create_view< boost::mpl::_ >, tile< 0, 0, 0 >, tile< 0, 0, 0 > > > >::
            type complete_list;
        // filter the list
        typedef
            typename boost::mpl::filter_view< complete_list, is_storage_wrapper< boost::mpl::_1 > >::type filtered_list;
        typedef typename boost::mpl::fold< filtered_list,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 > >::type type;
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

               \tparam The type element of a metadata set which is a pointer to a metadata
               \param mde The element of a metadata set which is a pointer to a metadata
             */
            template < typename MetaDataElem >
            bool operator()(MetaDataElem const *mde) const {
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
                if (MetaDataElem::layout_t::template at_< GridTraits::dim_k_t::value >::value >= 0) {
                    result = result && (grid.k_max() + 1 <= mde->template dim< GridTraits::dim_k_t::value >());
                }

                if (MetaDataElem::layout_t::template at_< GridTraits::dim_j_t::value >::value >= 0) {
                    result = result && (grid.j_high_bound() + 1 <= mde->template dim< GridTraits::dim_j_t::value >());
                }

                if (MetaDataElem::layout_t::template at_< GridTraits::dim_i_t::value >::value >= 0) {
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
        auto metadata_view = _impl::get_storage_info_ptrs(aggr.get_arg_storage_pairs());
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
        typename MssDescriptorForest,
        typename DomainType,
        typename Grid,
        bool IsStateful,
        uint_t RepeatFunctor = 1 >
    struct intermediate
        : public computation< DomainType,
              typename _impl::get_reduction_type< typename boost::mpl::back< MssDescriptorForest >::type >::type > {

        GRIDTOOLS_STATIC_ASSERT((is_condition_forest_of< MssDescriptorForest, is_computation_token >::value),
            "make_computation args should be mss descriptors or condition trees of mss descriptors");

        using branch_selector_t = branch_selector< MssDescriptorForest >;
        using branches_t = typename branch_selector_t::branches_t;

        GRIDTOOLS_STATIC_ASSERT(
            (copy_into_variadic_t< MssDescriptorForest, _impl::all_args_in_aggregator< DomainType > >::type::value),
            "Some placeholders used in the computation are not listed in the aggregator");

        using MssDescriptors =
            typename copy_into_variadic_t< MssDescriptorForest, meta_array_generator< boost::mpl::vector0<> > >::type;

        GRIDTOOLS_STATIC_ASSERT(
            (is_condition_tree_of_sequence_of< MssDescriptors, is_computation_token >::value), GT_INTERNAL_ERROR);

        GRIDTOOLS_STATIC_ASSERT((is_backend< Backend >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< DomainType >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);

        typedef typename Backend::backend_traits_t::performance_meter_t performance_meter_t;
        typedef typename Backend::backend_ids_t backend_ids_t;
        typedef grid_traits_from_id< backend_ids_t::s_grid_type_id > grid_traits_t;
        typedef typename DomainType::placeholders_t placeholders_t;

        // First we need to compute the association between placeholders and extents.
        // This information is needed to allocate temporaries, and to provide the
        // extent information to the user.
        typedef typename placeholder_to_extent_map< MssDescriptors, grid_traits_t, placeholders_t >::type extent_map_t;
        // Second we need to associate an extent to each esf, so that
        // we can associate loop bounds to the functors.
        typedef typename associate_extents_to_esfs< MssDescriptors, extent_map_t >::type extent_sizes_t;

        typedef typename boost::mpl::if_<
            boost::mpl::is_sequence< MssDescriptors >,
            typename boost::mpl::fold< MssDescriptors,
                boost::mpl::false_,
                boost::mpl::or_< boost::mpl::_1, mss_descriptor_is_reduction< boost::mpl::_2 > > >::type,
            boost::mpl::false_ >::type has_reduction_t;

        typedef reduction_data< MssDescriptors, has_reduction_t::value > reduction_data_t;
        typedef typename reduction_data_t::reduction_type_t reduction_type_t;

        typedef typename build_mss_components_array< backend_id< Backend >::value,
            MssDescriptors,
            extent_sizes_t,
            static_int< RepeatFunctor >,
            typename Grid::axis_type >::type mss_components_array_t;

        // creates a fusion sequence of views
        typedef typename create_view_fusion_map< DomainType >::type view_list_fusion_t;

        // create storage_wrapper_list
        typedef typename create_storage_wrapper_list< Backend, DomainType, mss_components_array_t >::type
            storage_wrapper_list_t;

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

        Grid m_grid;

        performance_meter_t m_meter;

        branch_selector< MssDescriptorForest > m_branch_selector;
        view_list_fusion_t m_view_list;
        storage_wrapper_fusion_list_t m_storage_wrapper_list;

        using intermediate::computation::m_domain;

        template < typename MssDescs >
        using convert_to_mss_components_t = typename build_mss_components_array< backend_id< Backend >::value,
            MssDescs,
            typename associate_extents_to_esfs< MssDescs, extent_map_t >::type,
            static_int< RepeatFunctor >,
            typename Grid::axis_type >::type;

        struct run_f {
            template < typename MssDescs >
            reduction_type_t operator()(MssDescs const &mss_descriptors,
                Grid const &grid,
                mss_local_domain_list_t const &mss_local_domain_list) const {
                reduction_data_t reduction_data(_impl::extract_reduction_intial_value_f{}(
                    boost::fusion::at_c< boost::mpl::size< MssDescriptorForest >::value - 1 >(mss_descriptors)));
                Backend::template run< convert_to_mss_components_t< MssDescs > >(
                    grid, mss_local_domain_list, reduction_data);
                return reduction_data.reduced_value();
            }
        };

      public:
        template < typename Domain, typename Forest >
        intermediate(Domain &&domain, Grid const &grid, Forest &&forest)
            : intermediate::computation(std::forward< Domain >(domain)), m_grid(grid), m_meter("NoName"),
              m_branch_selector(forest) {
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
            _impl::instantiate_views< Backend >(m_domain.get_arg_storage_pairs(), m_view_list);
            // fill storage_wrapper_list
            boost::fusion::for_each(
                m_storage_wrapper_list, _impl::initialize_storage_wrappers< view_list_fusion_t >(m_view_list));
            // setup the computation for given backend (e.g., move grid to device)
            Backend::setup_grid(m_grid);
            // instantiate mss_local_domains and wrapped local_domains with the right view_wrappers
            boost::fusion::for_each(m_mss_local_domain_list,
                _impl::instantiate_mss_local_domain< Backend, storage_wrapper_fusion_list_t, DomainType, IsStateful >(
                                        m_storage_wrapper_list, m_domain));
        }

        virtual void finalize() {
            // sync the data stores that should be synced
            boost::fusion::for_each(m_domain.get_arg_storage_pairs(), _impl::sync_data_stores());
        }

        virtual reduction_type_t run() {
            // check if all views are still consistent, otherwise we have to call steady again
            _impl::check_view_consistency< DomainType > check_views(m_domain);
            boost::fusion::for_each(m_view_list, check_views);
            if (!check_views.is_consistent()) {
                steady();
            }

            m_meter.start();
            auto res = m_branch_selector.apply(run_f{}, std::cref(m_grid), std::cref(m_mss_local_domain_list));
            m_meter.pause();
            return res;
        }

        virtual std::string print_meter() { return m_meter.to_string(); }

        virtual double get_meter() { return m_meter.total_time(); }

        virtual void reset_meter() { m_meter.reset(); }

        mss_local_domain_list_t const &mss_local_domain_list() const { return m_mss_local_domain_list; }

        // TODO(anstaf): This accessor breaks encapsulation and needed only for intermedite_expand implementation.
        //               Refactor ASAP.
        DomainType &domain() { return m_domain; }
        const DomainType &domain() const { return m_domain; }
    };

    template < typename T >
    struct intermediate_mss_local_domains;

    template < typename Backend,
        typename MssArray,
        typename DomainType,
        typename Grid,
        bool IsStateful,
        uint_t RepeatFunctor >
    struct intermediate_mss_local_domains<
        intermediate< Backend, MssArray, DomainType, Grid, IsStateful, RepeatFunctor > > {
        using type = typename intermediate< Backend, MssArray, DomainType, Grid, IsStateful, RepeatFunctor >::
            mss_local_domains_t;
    };
} // namespace gridtools

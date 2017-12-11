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

#include "amss_descriptor.hpp"
#include "backend_base.hpp"
#include "backend_metafunctions.hpp"
#include "backend_traits_fwd.hpp"
#include "computation.hpp"
#include "compute_extents_metafunctions.hpp"
#include "conditionals/condition_tree.hpp"
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

#include "computation_grammar.hpp"
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

    namespace _impl {
        struct dummy_run_f {
            template < typename Branch >
            reduction_type< Branch > operator()(Branch const &) const;
        };

        template < typename... MssDescriptors >
        using reduction_type_from_forest_t =
            decltype(std::declval< branch_selector< MssDescriptors... > >().apply(dummy_run_f{}));
    }

    /**
     * @class
     *  @brief structure collecting helper metafunctions
     */

    template < uint_t RepeatFunctor,
        bool IsStateful,
        typename Backend,
        typename DomainType,
        typename Grid,
        typename... MssDescriptors >
    struct intermediate : public computation< DomainType, _impl::reduction_type_from_forest_t< MssDescriptors... > > {

        GRIDTOOLS_STATIC_ASSERT((is_backend< Backend >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< DomainType >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::and_< std::true_type,
                                    is_condition_tree_of< MssDescriptors, is_computation_token >... >::value),
            "make_computation args should be mss descriptors or condition trees of mss descriptors");
        GRIDTOOLS_STATIC_ASSERT((_impl::all_args_in_aggregator< DomainType, MssDescriptors... >::type::value),
            "Some placeholders used in the computation are not listed in the aggregator");

        using branch_selector_t = branch_selector< MssDescriptors... >;
        using all_mss_descriptors_t = typename branch_selector_t::all_leaves_t;

        typedef typename Backend::backend_traits_t::performance_meter_t performance_meter_t;
        typedef typename Backend::grid_traits_t grid_traits_t;
        typedef typename DomainType::placeholders_t placeholders_t;

        // First we need to compute the association between placeholders and extents.
        // This information is needed to allocate temporaries, and to provide the
        // extent information to the user.
        using extent_map_t =
            typename placeholder_to_extent_map< all_mss_descriptors_t, grid_traits_t, placeholders_t >::type;

        template < typename MssDescs >
        using convert_to_mss_components_array_t =
            typename build_mss_components_array< typename Backend::mss_fuse_esfs_strategy,
                MssDescs,
                extent_map_t,
                static_int< RepeatFunctor >,
                typename Grid::axis_type >::type;

        typedef convert_to_mss_components_array_t< all_mss_descriptors_t > mss_components_array_t;

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
        typedef typename boost::fusion::result_of::as_vector< mss_local_domains_t >::type mss_local_domain_list_t;

        // member fields
        mss_local_domain_list_t m_mss_local_domain_list;

        Grid m_grid;

        performance_meter_t m_meter;

        branch_selector_t m_branch_selector;
        view_list_fusion_t m_view_list;
        storage_wrapper_fusion_list_t m_storage_wrapper_list;

        using intermediate::computation::m_domain;

        struct run_f {
            template < typename MssDescs >
            reduction_type< MssDescs > operator()(MssDescs const &mss_descriptors,
                Grid const &grid,
                mss_local_domain_list_t const &mss_local_domain_list) const {
                auto reduction_data = make_reduction_data(mss_descriptors);
                Backend::template run< convert_to_mss_components_array_t< MssDescs > >(
                    grid, mss_local_domain_list, reduction_data);
                return reduction_data.reduced_value();
            }
        };

      public:
        template < typename Domain, typename... Msses >
        intermediate(Domain &&domain, Grid const &grid, Msses &&... msses)
            : intermediate::computation(std::forward< Domain >(domain)), m_grid(grid), m_meter("NoName"),
              m_branch_selector(std::forward< Msses >(msses)...) {
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
            typename Backend::setup_grid_f{}(m_grid);
            // instantiate mss_local_domains and wrapped local_domains with the right view_wrappers
            boost::fusion::for_each(m_mss_local_domain_list,
                _impl::instantiate_mss_local_domain< Backend, storage_wrapper_fusion_list_t, DomainType, IsStateful >(
                                        m_storage_wrapper_list, m_domain));
        }

        virtual void finalize() {
            // sync the data stores that should be synced
            boost::fusion::for_each(m_domain.get_arg_storage_pairs(), _impl::sync_data_stores());
        }

        virtual typename intermediate::return_t run() {
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
    struct intermediate_mss_local_domains {
        using type = typename T::mss_local_domains_t;
    };
} // namespace gridtools

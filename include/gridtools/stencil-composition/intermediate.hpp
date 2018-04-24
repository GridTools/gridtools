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

#include <tuple>
#include <utility>

#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/std_tuple.hpp>
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

#include "backend_base.hpp"
#include "backend_metafunctions.hpp"
#include "backend_traits_fwd.hpp"
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
#include "iterate_on_esfs.hpp"
#include "extract_placeholders.hpp"

#include "../common/generic_metafunctions/meta.hpp"
#include "../common/tuple_util.hpp"

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
        using type = typename _impl::get_view< typename get_data_store_from_arg< Placeholder >::type >::type;
    };

    template < typename Backend, typename Placeholders, typename MssComponentsArray >
    struct create_storage_wrapper_list {
        // handle all tmps, obtain the storage_wrapper_list for written tmps
        typedef
            typename _impl::obtain_storage_wrapper_list_t< Backend, Placeholders, MssComponentsArray >::type all_tmps;

        // for every placeholder we push back an element that is either a new storage_wrapper type
        // for a normal data_store(_field), or in case it is a tmp we get the element out of the all_tmps list.
        // if we find a read-only tmp void will be pushed back, but this will be filtered out in the
        // last step.
        typedef typename boost::mpl::transform_view<
            Placeholders,
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

    template < typename MssDescs >
    struct need_to_compute_extents {

        /* helper since boost::mpl::and_ fails in this case with nvcc
        */
        template < typename BoolA, typename BoolB >
        struct gt_and : std::integral_constant< bool, BoolA::value and BoolB::value > {};

        /* helper since boost::mpl::or_ fails in this case with nvcc
        */
        template < typename BoolA, typename BoolB >
        struct gt_or : std::integral_constant< bool, BoolA::value or BoolB::value > {};

        using has_all_extents = typename with_operators< is_esf_with_extent,
            gt_and >::template iterate_on_esfs< std::true_type, MssDescs >::type;
        using has_extent = typename with_operators< is_esf_with_extent,
            gt_or >::template iterate_on_esfs< std::false_type, MssDescs >::type;

        GRIDTOOLS_STATIC_ASSERT((has_extent::value == has_all_extents::value),
            "The computation appears to have stages with and without extents being specified at the same time. A "
            "computation should have all stages with extents or none.");
        using type = typename boost::mpl::not_< has_all_extents >::type;
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

        template < int I, class Layout >
        using exists_in_layout = bool_constant < I< Layout::masked_length >;

        template < int I, uint_t Id, class Layout, class Halo, class Alignment >
        enable_if_t< exists_in_layout< I, Layout >::value, bool > storage_info_dim_fits(
            storage_info_interface< Id, Layout, Halo, Alignment > const &storage_info, int val) {
            return val + 1 <= storage_info.template dim< I >();
        }
        template < int I, uint_t Id, class Layout, class Halo, class Alignment >
        enable_if_t< !exists_in_layout< I, Layout >::value, bool > storage_info_dim_fits(
            storage_info_interface< Id, Layout, Halo, Alignment > const &, int) {
            return true;
        }

        template < class GridTraits, class Grid >
        struct storage_info_fits_grid_f {
            Grid const &grid;

            template < uint_t Id, class Layout, class Halo, class Alignment >
            bool operator()(storage_info_interface< Id, Layout, Halo, Alignment > const &src) const {

                // TODO: This check may be not accurate since there is
                // an ongoing change in the convention for storage and
                // grid. Before the storage had the conventions that
                // there was not distinction between halo and core
                // region in the storage. The distinction was made
                // solely in the grid. Now the storage makes that
                // distinction, ad when allocating the data the halo
                // is also allocated. So for instance a storage of
                // 3x3x3 with halo of <1,1,1> will allocate a 5x5x5
                // storage. The grid is the same as before. The first
                // step will be to update the storage to point as
                // first element the (1,1,1) element and then to get
                // the grid to not specifying halos (at least in the
                // simple cases). This is why the check is left as
                // before here, but may be updated with more accurate
                // ones when the convention is updated
                return storage_info_dim_fits< GridTraits::dim_k_t::value >(src, grid.k_max()) &&
                       storage_info_dim_fits< GridTraits::dim_j_t::value >(src, grid.j_high_bound()) &&
                       storage_info_dim_fits< GridTraits::dim_i_t::value >(src, grid.i_high_bound());
            }
        };

    } // namespace _impl

    /**
     *   This functor checks that grid size is small enough to not make the stencil go out of bound on data fields.
     *
     *   \tparam GridTraits The grid traits of the grid in question to get the indices of relevant coordinates
     *   \tparam Grid The Grid
     */
    template < class GridTraits, class Grid >
    _impl::storage_info_fits_grid_f< GridTraits, Grid > storage_info_fits_grid(Grid const &grid) {
        return {grid};
    }

    namespace _impl {
        struct dummy_run_f {
            template < typename T >
            reduction_type< T > operator()(T const &) const;
        };
    }

    /**
     *  @brief structure collecting helper metafunctions
     */
    template < uint_t RepeatFunctor,
        bool IsStateful,
        class Backend,
        class Grid,
        class BoundArgStoragePairs,
        class MssDescriptors >
    class intermediate;

    template < uint_t RepeatFunctor,
        bool IsStateful,
        class Backend,
        class Grid,
        class... BoundPlaceholders,
        class... BoundDataStores,
        class... MssDescriptors >
    class intermediate< RepeatFunctor,
        IsStateful,
        Backend,
        Grid,
        std::tuple< arg_storage_pair< BoundPlaceholders, BoundDataStores >... >,
        std::tuple< MssDescriptors... > > {
        GRIDTOOLS_STATIC_ASSERT((is_backend< Backend >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);

        GRIDTOOLS_STATIC_ASSERT((conjunction< is_condition_tree_of< MssDescriptors, is_computation_token >... >::value),
            "make_computation args should be mss descriptors or condition trees of mss descriptors");

        using branch_selector_t = branch_selector< MssDescriptors... >;
        using all_mss_descriptors_t = typename branch_selector_t::all_leaves_t;
        using return_type = decltype(std::declval< branch_selector_t >().apply(_impl::dummy_run_f{}));

        typedef typename Backend::backend_traits_t::performance_meter_t performance_meter_t;
        typedef typename Backend::grid_traits_t grid_traits_t;

        using placeholders_t = GT_META_CALL(extract_placeholders, all_mss_descriptors_t);
        using tmp_placeholders_t = GT_META_CALL(meta::filter, (is_tmp_arg, placeholders_t));
        using non_tmp_placeholders_t = GT_META_CALL(meta::filter, (meta::not_< is_tmp_arg >::apply, placeholders_t));

#if GT_BROKEN_TEMPLATE_ALIASES
        template < class Arg >
        struct to_arg_storage_pair {
            using type = arg_storage_pair< Arg, typename Arg::data_store_t >;
        };
        using tmp_arg_storage_pair_tuple_t = GT_META_CALL(
            meta::rename,
            (meta::defer< std::tuple >::apply,
                GT_META_CALL(meta::transform, (to_arg_storage_pair, tmp_placeholders_t))));
#else
        template < class Arg >
        using to_arg_storage_pair = arg_storage_pair< Arg, typename Arg::data_store_t >;

        using tmp_arg_storage_pair_tuple_t =
            meta::rename< std::tuple, meta::transform< to_arg_storage_pair, tmp_placeholders_t > >;
#endif

        GRIDTOOLS_STATIC_ASSERT(
            (conjunction< meta::st_contains< non_tmp_placeholders_t, BoundPlaceholders >... >::value),
            "some bound placeholders are not used in mss descriptors");

        GRIDTOOLS_STATIC_ASSERT(meta::is_set_fast< meta::list< BoundPlaceholders... > >::value,
            "bound placeholders should be all different");

        template < class Arg >
        using is_free = negation< meta::st_contains< meta::list< BoundPlaceholders... >, Arg > >;

        using free_placeholders_t = GT_META_CALL(meta::filter, (is_free, non_tmp_placeholders_t));

        using storage_info_map_t = _impl::storage_info_map_t< placeholders_t >;

        using bound_arg_storage_pair_tuple_t =
            std::tuple< _impl::bound_arg_storage_pair< BoundPlaceholders, BoundDataStores >... >;

      public:
        // First we need to compute the association between placeholders and extents.
        // This information is needed to allocate temporaries, and to provide the
        // extent information to the user.
        using extent_map_t =
            typename boost::mpl::eval_if< typename need_to_compute_extents< all_mss_descriptors_t >::type,
                placeholder_to_extent_map< all_mss_descriptors_t, grid_traits_t, placeholders_t >,
                boost::mpl::void_ >::type;

      private:
        template < typename MssDescs >
        using convert_to_mss_components_array_t =
            typename build_mss_components_array< typename Backend::mss_fuse_esfs_strategy,
                MssDescs,
                extent_map_t,
                static_int< RepeatFunctor >,
                typename Grid::axis_type >::type;

        typedef convert_to_mss_components_array_t< all_mss_descriptors_t > mss_components_array_t;

        // create storage_wrapper_list
        typedef typename create_storage_wrapper_list< Backend, placeholders_t, mss_components_array_t >::type
            storage_wrapper_list_t;

        // get the maximum extent (used to retrieve the size of the temporaries)
        typedef typename max_i_extent_from_storage_wrapper_list< storage_wrapper_list_t >::type max_i_extent_t;

      public:
        // creates an mpl sequence of local domains
        typedef typename create_mss_local_domains< backend_id< Backend >::value,
            mss_components_array_t,
            storage_wrapper_list_t,
            IsStateful >::type mss_local_domains_t;

      private:
        // creates a tuple of local domains
        using mss_local_domain_list_t = copy_into_variadic< mss_local_domains_t, std::tuple<> >;

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

        // member fields

        Grid m_grid;

        performance_meter_t m_meter;

        /// branch_selector is responsible for choosing the right branch of in condition MSS tree.
        //
        branch_selector_t m_branch_selector;

        /// is needed for dedup_storage_info method.
        //
        storage_info_map_t m_storage_info_map;

        /// tuple with temporary storages
        //
        tmp_arg_storage_pair_tuple_t m_tmp_arg_storage_pair_tuple;

        /// tuple with storages that are bound during costruction
        //  Each item holds a storage and its view
        bound_arg_storage_pair_tuple_t m_bound_arg_storage_pair_tuple;

        /// Here are local domains (structures with raw pointers for passing to backed.
        //
        mss_local_domain_list_t m_mss_local_domain_list;

        bool m_timer_enabled;

      public:
        intermediate(Grid const &grid,
            std::tuple< arg_storage_pair< BoundPlaceholders, BoundDataStores >... > arg_storage_pairs,
            std::tuple< MssDescriptors... > msses,
            bool timer_enabled = true)
            // grid just stored to the member
            : m_grid(grid),
              m_meter("NoName"),
              // pass mss descriptor condition trees to branch_selector that owns them and provides the interface to
              // a functor with a chosen condition branch
              m_branch_selector(std::move(msses)),
              // here we create temporary storages; note that they are passed through the `dedup_storage_info` method.
              // that ensures, that only
              m_tmp_arg_storage_pair_tuple(dedup_storage_info(_impl::make_tmp_arg_storage_pairs< max_i_extent_t,
                  Backend,
                  storage_wrapper_list_t,
                  tmp_arg_storage_pair_tuple_t >(grid))),
              // stash bound storages; sanitizing them through the `dedup_storage_info` as well.
              m_bound_arg_storage_pair_tuple(dedup_storage_info(std::move(arg_storage_pairs))),
              m_timer_enabled(timer_enabled) {

            // check_grid_against_extents< all_extents_vecs_t >(grid);
            // check_fields_sizes< grid_traits_t >(grid, domain);

            // Here we make views (actually supplemental view_info structures are made) from both temporary and bound
            // storages, concatenate them together and pass to `update_local_domains`
            update_local_domains(std::tuple_cat(
                make_view_infos(m_tmp_arg_storage_pair_tuple), make_view_infos(m_bound_arg_storage_pair_tuple)));
            // now only local domanis missing pointers from free (not bound) storages.
        }

        void sync_bound_data_stores() const { tuple_util::for_each(_impl::sync_f{}, m_bound_arg_storage_pair_tuple); }

        // TODO(anstaf): introduce overload that takes a tuple of arg_storage_pair's. it will simplify a bit
        //               implementation of the `intermediate_expanded` and `computation` by getting rid of
        //               `boost::fusion::invoke`.
        template < class... Args, class... DataStores >
        typename std::enable_if< sizeof...(Args) == meta::length< free_placeholders_t >::value, return_type >::type run(
            arg_storage_pair< Args, DataStores > const &... srcs) {
            GRIDTOOLS_STATIC_ASSERT((conjunction< meta::st_contains< free_placeholders_t, Args >... >::value),
                "some placeholders are not used in mss descriptors");
            GRIDTOOLS_STATIC_ASSERT(
                meta::is_set_fast< meta::list< Args... > >::value, "free placeholders should be all different");

            // make views from bound storages again (in the case old views got inconsistent)
            // make views from free storages;
            // concatenate them into a single tuple.
            // push view for updating the local domains.
            update_local_domains(std::tuple_cat(make_view_infos(m_bound_arg_storage_pair_tuple),
                make_view_infos(dedup_storage_info(std::tie(srcs...)))));
            // now local domains are fully set up.
            if (m_timer_enabled)
                m_meter.start();
            // branch selector calls run_f functor on the right branch of mss condition tree.
            auto res = m_branch_selector.apply(run_f{}, std::cref(m_grid), std::cref(m_mss_local_domain_list));
            if (m_timer_enabled)
                m_meter.pause();
            return res;
        }

        std::string print_meter() const { return m_meter.to_string(); }

        double get_meter() const { return m_meter.total_time(); }

        void reset_meter() { m_meter.reset(); }

        mss_local_domain_list_t const &mss_local_domain_list() const { return m_mss_local_domain_list; }

      private:
        template < class Src >
        static auto make_view_infos(Src &&src)
            GT_AUTO_RETURN(tuple_util::transform(_impl::make_view_info_f< Backend >{}, std::forward< Src >(src)));

        template < class Views >
        void update_local_domains(Views const &views) {
            _impl::update_local_domains(views, m_mss_local_domain_list);
        }

        template < class Seq >
        auto dedup_storage_info(Seq const &seq) GT_AUTO_RETURN(
            tuple_util::transform(_impl::dedup_storage_info_f< storage_info_map_t >{m_storage_info_map}, seq));
    };

    /**
     *  This metafunction exposes intermediate implementation specific details to a couple of unit tests.
     *
     *  @todo(anstaf):
     *  I would consider this as a design flaw. `mss_local_domains_t` is not logically the interface of `intermediate`.
     *  Probably the creation of local domains should be factored out into a separate component to resolve this issue.
     */
    template < typename Intermediate >
    using intermediate_mss_local_domains = typename Intermediate::mss_local_domains_t;

} // namespace gridtools

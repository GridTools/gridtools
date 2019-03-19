/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
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
#include <boost/mpl/max_element.hpp>
#include <boost/mpl/min_element.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/type_traits/remove_const.hpp>

#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "backend_base.hpp"
#include "backend_metafunctions.hpp"
#include "backend_traits_fwd.hpp"
#include "compute_extents_metafunctions.hpp"
#include "conditionals/condition_tree.hpp"
#include "coordinate.hpp"
#include "esf.hpp"
#include "extract_placeholders.hpp"
#include "grid.hpp"
#include "grid_traits.hpp"
#include "intermediate_impl.hpp"
#include "level.hpp"
#include "local_domain.hpp"
#include "mss_components_metafunctions.hpp"

/**
 * @file
 * \brief this file contains mainly helper metafunctions which simplify the interface for the application developer
 * */
namespace gridtools {
    namespace _impl {

        template <int I, class Layout>
        using exists_in_layout = bool_constant < I<Layout::masked_length>;

        template <int I, uint_t Id, class Layout, class Halo, class Alignment>
        enable_if_t<exists_in_layout<I, Layout>::value, bool> storage_info_dim_fits(
            storage_info<Id, Layout, Halo, Alignment> const &storage_info, int val) {
            return val + 1 <= storage_info.template total_length<I>();
        }
        template <int I, uint_t Id, class Layout, class Halo, class Alignment>
        enable_if_t<!exists_in_layout<I, Layout>::value, bool> storage_info_dim_fits(
            storage_info<Id, Layout, Halo, Alignment> const &, int) {
            return true;
        }

        template <class Backend, class Grid>
        struct storage_info_fits_grid_f {
            Grid const &grid;

            template <uint_t Id, class Layout, class Halo, class Alignment>
            bool operator()(storage_info<Id, Layout, Halo, Alignment> const &src) const {

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
                return storage_info_dim_fits<coord_k<Backend>::value>(src, grid.k_max()) &&
                       storage_info_dim_fits<coord_j<Backend>::value>(src, grid.j_high_bound()) &&
                       storage_info_dim_fits<coord_i<Backend>::value>(src, grid.i_high_bound());
            }
        };

        template <class Mss>
        GT_META_DEFINE_ALIAS(get_esfs, unwrap_independent, typename Mss::esf_sequence_t);

    } // namespace _impl

    /**
     *   This functor checks that grid size is small enough to not make the stencil go out of bound on data fields.
     *
     *   \tparam GridTraits The grid traits of the grid in question to get the indices of relevant coordinates
     *   \tparam Grid The Grid
     */
    template <class BackendIds, class Grid>
    _impl::storage_info_fits_grid_f<BackendIds, Grid> storage_info_fits_grid(Grid const &grid) {
        return {grid};
    }

    /**
     *  @brief structure collecting helper metafunctions
     */
    template <bool IsStateful, class Backend, class Grid, class BoundArgStoragePairs, class MssDescriptors>
    class intermediate;

    template <bool IsStateful,
        class Backend,
        class Grid,
        class... BoundPlaceholders,
        class... BoundDataStores,
        class... MssDescriptors>
    class intermediate<IsStateful,
        Backend,
        Grid,
        std::tuple<arg_storage_pair<BoundPlaceholders, BoundDataStores>...>,
        std::tuple<MssDescriptors...>> {
        GT_STATIC_ASSERT(is_backend<Backend>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);

        GT_STATIC_ASSERT((conjunction<is_condition_tree_of<MssDescriptors, is_mss_descriptor>...>::value),
            "make_computation args should be mss descriptors or condition trees of mss descriptors");

        using branch_selector_t = branch_selector<MssDescriptors...>;
        using all_mss_descriptors_t = typename branch_selector_t::all_leaves_t;

        typedef typename Backend::backend_traits_t::performance_meter_t performance_meter_t;

        using placeholders_t = GT_META_CALL(extract_placeholders_from_msses, all_mss_descriptors_t);
        using tmp_placeholders_t = GT_META_CALL(meta::filter, (is_tmp_arg, placeholders_t));
        using non_tmp_placeholders_t = GT_META_CALL(meta::filter, (meta::not_<is_tmp_arg>::apply, placeholders_t));

        using non_cached_tmp_placeholders_t = GT_META_CALL(
            _impl::extract_non_cached_tmp_args_from_msses, all_mss_descriptors_t);

        template <class Arg>
        GT_META_DEFINE_ALIAS(to_arg_storage_pair, meta::id, (arg_storage_pair<Arg, typename Arg::data_store_t>));

        using tmp_arg_storage_pair_tuple_t = GT_META_CALL(meta::transform,
            (to_arg_storage_pair,
                GT_META_CALL(meta::if_,
                    (GT_META_CALL(needs_allocate_cached_tmp, Backend),
                        tmp_placeholders_t,
                        non_cached_tmp_placeholders_t))));

        GT_STATIC_ASSERT((conjunction<meta::st_contains<non_tmp_placeholders_t, BoundPlaceholders>...>::value),
            "some bound placeholders are not used in mss descriptors");

        GT_STATIC_ASSERT(
            meta::is_set_fast<meta::list<BoundPlaceholders...>>::value, "bound placeholders should be all different");

        template <class Arg>
        using is_free = negation<meta::st_contains<meta::list<BoundPlaceholders...>, Arg>>;

        using free_placeholders_t = GT_META_CALL(meta::filter, (is_free, non_tmp_placeholders_t));

        using storage_info_map_t = _impl::storage_info_map_t<placeholders_t>;

        using bound_arg_storage_pair_tuple_t = std::tuple<arg_storage_pair<BoundPlaceholders, BoundDataStores>...>;

        using esfs_t = GT_META_CALL(
            meta::flatten, (GT_META_CALL(meta::transform, (_impl::get_esfs, all_mss_descriptors_t))));

      public:
        // First we need to compute the association between placeholders and extents.
        // This information is needed to allocate temporaries, and to provide the extent information to the user.
        using extent_map_t = GT_META_CALL(get_extent_map, esfs_t);

      private:
        template <class MssDescs>
        GT_META_DEFINE_ALIAS(convert_to_mss_components_array,
            build_mss_components_array,
            (Backend::mss_fuse_esfs_strategy::value, MssDescs, extent_map_t, typename Grid::axis_type));

        using mss_components_array_t = GT_META_CALL(convert_to_mss_components_array, all_mss_descriptors_t);

        using max_extent_for_tmp_t = GT_META_CALL(_impl::get_max_extent_for_tmp, mss_components_array_t);

      public:
        // creates a tuple of local domains
        using local_domains_t = GT_META_CALL(_impl::get_local_domains, (mss_components_array_t, IsStateful));

      private:
        struct run_f {
            template <typename MssDescs>
            void operator()(
                MssDescs const &mss_descriptors, Grid const &grid, local_domains_t const &local_domains) const {
                Backend::template run<GT_META_CALL(convert_to_mss_components_array, MssDescs)>(grid, local_domains);
            }
        };

        // member fields

        Grid m_grid;

        std::unique_ptr<performance_meter_t> m_meter;

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

        /// Here are local domains (structures with raw pointers for passing to backend.
        //
        local_domains_t m_local_domains;

        struct check_grid_against_extents_f {
            Grid const &m_grid;

            template <class Placeholder>
            void operator()() const {
                using extent_t = decltype(intermediate::get_arg_extent(Placeholder()));
                assert(-extent_t::iminus::value <= static_cast<int_t>(m_grid.direction_i().minus()));
                assert(extent_t::iplus::value <= static_cast<int_t>(m_grid.direction_i().plus()));
                assert(-extent_t::jminus::value <= static_cast<int_t>(m_grid.direction_j().minus()));
                assert(extent_t::jplus::value <= static_cast<int_t>(m_grid.direction_j().plus()));
            }
        };

      public:
        intermediate(Grid const &grid,
            std::tuple<arg_storage_pair<BoundPlaceholders, BoundDataStores>...> arg_storage_pairs,
            std::tuple<MssDescriptors...> msses,
            bool timer_enabled = true)
            // grid just stored to the member
            : m_grid(grid),
              // pass mss descriptor condition trees to branch_selector that owns them and provides the interface to
              // a functor with a chosen condition branch
              m_branch_selector(std::move(msses)),
              // here we create temporary storages; note that they are passed through the `dedup_storage_info` method.
              m_tmp_arg_storage_pair_tuple(dedup_storage_info(
                  _impl::make_tmp_arg_storage_pairs<max_extent_for_tmp_t, Backend, tmp_arg_storage_pair_tuple_t>(
                      grid))),
              // stash bound storages; sanitizing them through the `dedup_storage_info` as well.
              m_bound_arg_storage_pair_tuple(dedup_storage_info(std::move(arg_storage_pairs))) {
            if (timer_enabled)
                m_meter.reset(new performance_meter_t{"NoName"});
#ifndef NDEBUG
            check_grid_against_extents();
#endif
        }

        // TODO(anstaf): introduce overload that takes a tuple of arg_storage_pair's. it will simplify a bit
        //               implementation of the `intermediate_expanded` and `computation` by getting rid of
        //               `boost::fusion::invoke`.
        template <class... Args, class... DataStores>
        enable_if_t<sizeof...(Args) == meta::length<free_placeholders_t>::value> run(
            arg_storage_pair<Args, DataStores> const &... srcs) {
            if (m_meter)
                m_meter->start();
            GT_STATIC_ASSERT((conjunction<meta::st_contains<free_placeholders_t, Args>...>::value),
                "some placeholders are not used in mss descriptors");
            GT_STATIC_ASSERT(
                meta::is_set_fast<meta::list<Args...>>::value, "free placeholders should be all different");
            m_branch_selector.apply(run_f{}, std::cref(m_grid), local_domains(srcs...));
            if (m_meter)
                m_meter->pause();
        }

        std::string print_meter() const {
            assert(m_meter);
            return m_meter->to_string();
        }

        double get_time() const {
            assert(m_meter);
            return m_meter->total_time();
        }

        size_t get_count() const {
            assert(m_meter);
            return m_meter->count();
        }

        void reset_meter() {
            assert(m_meter);
            m_meter->reset();
        }

        template <class Placeholder,
            class RwArgs = GT_META_CALL(_impl::all_rw_args, all_mss_descriptors_t),
            intent Intent = meta::st_contains<RwArgs, Placeholder>::value ? intent::inout : intent::in>
        static constexpr std::integral_constant<intent, Intent> get_arg_intent(Placeholder) {
            return {};
        }

        template <class Placeholder>
        static constexpr GT_META_CALL(lookup_extent_map, (extent_map_t, Placeholder)) get_arg_extent(Placeholder) {
            GT_STATIC_ASSERT(is_plh<Placeholder>::value, "");
            return {};
        }

        template <class... Args, class... DataStores>
        local_domains_t const &local_domains(arg_storage_pair<Args, DataStores> const &... srcs) {
            _impl::update_local_domains(tuple_util::flatten(std::make_tuple(m_tmp_arg_storage_pair_tuple,
                                            m_bound_arg_storage_pair_tuple,
                                            dedup_storage_info(std::tie(srcs...)))),
                m_local_domains);
            return m_local_domains;
        }

      private:
        template <class Seq>
        auto dedup_storage_info(Seq const &seq) GT_AUTO_RETURN(
            tuple_util::transform(_impl::dedup_storage_info_f<storage_info_map_t>{m_storage_info_map}, seq));

        void check_grid_against_extents() const {
            for_each_type<non_tmp_placeholders_t>(check_grid_against_extents_f{m_grid});
        }
    }; // namespace gridtools

    /**
     *  This metafunction exposes intermediate implementation specific details to a couple of unit tests.
     *
     *  @todo(anstaf):
     *  I would consider this as a design flaw. `mss_local_domains_t` is not logically the interface of `intermediate`.
     *  Probably the creation of local domains should be factored out into a separate component to resolve this issue.
     */
    template <typename Intermediate>
    using intermediate_local_domains = typename Intermediate::local_domains_t;

} // namespace gridtools

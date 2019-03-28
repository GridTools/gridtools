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

#include "../common/timer/timer_traits.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "backend_base.hpp"
#include "compute_extents_metafunctions.hpp"
#include "dim.hpp"
#include "esf.hpp"
#include "extract_placeholders.hpp"
#include "fused_mss_loop.hpp"
#include "grid.hpp"
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

        template <class Target, class Grid>
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
                return storage_info_dim_fits<dim::k::value>(src, grid.k_max()) &&
                       storage_info_dim_fits<dim::j::value>(src, grid.j_high_bound()) &&
                       storage_info_dim_fits<dim::i::value>(src, grid.i_high_bound());
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
    template <class Target, class Grid>
    _impl::storage_info_fits_grid_f<Target, Grid> storage_info_fits_grid(Grid const &grid) {
        return {grid};
    }

    /**
     *  @brief structure collecting helper metafunctions
     */
    template <bool IsStateful, class Target, class Grid, class BoundArgStoragePairs, class MssDescriptors>
    class intermediate;

    template <bool IsStateful,
        class Target,
        class Grid,
        class... BoundPlaceholders,
        class... BoundDataStores,
        class... MssDescriptors>
    class intermediate<IsStateful,
        Target,
        Grid,
        std::tuple<arg_storage_pair<BoundPlaceholders, BoundDataStores>...>,
        std::tuple<MssDescriptors...>> {
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);

        GT_STATIC_ASSERT(conjunction<is_mss_descriptor<MssDescriptors>...>::value,
            "make_computation args should be mss descriptors");

        using mss_descriptors_t = std::tuple<MssDescriptors...>;

        using performance_meter_t = typename timer_traits<Target>::timer_type;

        using placeholders_t = GT_META_CALL(extract_placeholders_from_msses, mss_descriptors_t);
        using tmp_placeholders_t = GT_META_CALL(meta::filter, (is_tmp_arg, placeholders_t));
        using non_tmp_placeholders_t = GT_META_CALL(meta::filter, (meta::not_<is_tmp_arg>::apply, placeholders_t));

        using non_cached_tmp_placeholders_t = GT_META_CALL(
            _impl::extract_non_cached_tmp_args_from_msses, mss_descriptors_t);

        template <class Arg>
        GT_META_DEFINE_ALIAS(to_arg_storage_pair, meta::id, (arg_storage_pair<Arg, typename Arg::data_store_t>));

        using tmp_arg_storage_pair_tuple_t = GT_META_CALL(meta::transform,
            (to_arg_storage_pair,
                GT_META_CALL(meta::if_,
                    (GT_META_CALL(needs_allocate_cached_tmp, Target),
                        tmp_placeholders_t,
                        non_cached_tmp_placeholders_t))));

        GT_STATIC_ASSERT((conjunction<meta::st_contains<non_tmp_placeholders_t, BoundPlaceholders>...>::value),
            "some bound placeholders are not used in mss descriptors");

        GT_STATIC_ASSERT(
            meta::is_set_fast<meta::list<BoundPlaceholders...>>::value, "bound placeholders should be all different");

        template <class Arg>
        using is_free = negation<meta::st_contains<meta::list<BoundPlaceholders...>, Arg>>;

        using free_placeholders_t = GT_META_CALL(meta::filter, (is_free, non_tmp_placeholders_t));

        using bound_arg_storage_pair_tuple_t = std::tuple<arg_storage_pair<BoundPlaceholders, BoundDataStores>...>;

        using esfs_t = GT_META_CALL(
            meta::flatten, (GT_META_CALL(meta::transform, (_impl::get_esfs, mss_descriptors_t))));

      public:
        // First we need to compute the association between placeholders and extents.
        // This information is needed to allocate temporaries, and to provide the extent information to the user.
        using extent_map_t = GT_META_CALL(get_extent_map, esfs_t);

      private:
        using fuse_esfs_t = decltype(mss_fuse_esfs(std::declval<Target>()));
        using mss_components_array_t = GT_META_CALL(build_mss_components_array,
            (fuse_esfs_t::value, mss_descriptors_t, extent_map_t, typename Grid::axis_type));

        using max_extent_for_tmp_t = GT_META_CALL(_impl::get_max_extent_for_tmp, mss_components_array_t);

      public:
        // creates a tuple of local domains
        using local_domains_t = GT_META_CALL(_impl::get_local_domains, (mss_components_array_t, IsStateful));

      private:
        // member fields

        Grid m_grid;

        std::unique_ptr<performance_meter_t> m_meter;

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
            bool timer_enabled = true)
            // grid just stored to the member
            : m_grid(grid),
              // here we create temporary storages; note that they are passed through the `dedup_storage_info` method.
              m_tmp_arg_storage_pair_tuple(
                  _impl::make_tmp_arg_storage_pairs<max_extent_for_tmp_t, Target, tmp_arg_storage_pair_tuple_t>(grid)),
              // stash bound storages
              m_bound_arg_storage_pair_tuple(std::move(arg_storage_pairs)) {
            if (timer_enabled)
                m_meter.reset(new performance_meter_t{"NoName"});
#ifndef NDEBUG
            for_each_type<non_tmp_placeholders_t>(check_grid_against_extents_f{m_grid});
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
            static constexpr auto backend_target = Target{};
            fused_mss_loop<mss_components_array_t>(backend_target, local_domains(srcs...), m_grid);
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
            class RwArgs = GT_META_CALL(_impl::all_rw_args, mss_descriptors_t),
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
            _impl::update_local_domains(
                tuple_util::flatten(
                    std::make_tuple(m_tmp_arg_storage_pair_tuple, m_bound_arg_storage_pair_tuple, std::tie(srcs...))),
                m_local_domains);
            return m_local_domains;
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

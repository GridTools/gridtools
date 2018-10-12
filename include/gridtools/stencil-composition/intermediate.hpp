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

#include "backend_base.hpp"
#include "backend_metafunctions.hpp"
#include "backend_traits_fwd.hpp"
#include "compute_extents_metafunctions.hpp"
#include "conditionals/condition_tree.hpp"
#include "coordinate.hpp"
#include "esf.hpp"
#include "grid.hpp"
#include "grid_traits.hpp"
#include "intermediate_impl.hpp"
#include "level.hpp"
#include "local_domain.hpp"
#include "mss_components_metafunctions.hpp"
#include "reductions/reduction_data.hpp"

#include "computation_grammar.hpp"
#include "extract_placeholders.hpp"
#include "iterate_on_esfs.hpp"

#include "../common/generic_metafunctions/meta.hpp"
#include "../common/tuple_util.hpp"

/**
 * @file
 * \brief this file contains mainly helper metafunctions which simplify the interface for the application developer
 * */
namespace gridtools {

    template <typename MssDescs>
    struct need_to_compute_extents {

        /* helper since boost::mpl::and_ fails in this case with nvcc
         */
        template <typename BoolA, typename BoolB>
        struct gt_and : std::integral_constant<bool, BoolA::value and BoolB::value> {};

        /* helper since boost::mpl::or_ fails in this case with nvcc
         */
        template <typename BoolA, typename BoolB>
        struct gt_or : std::integral_constant<bool, BoolA::value or BoolB::value> {};

        using has_all_extents = typename with_operators<is_esf_with_extent,
            gt_and>::template iterate_on_esfs<std::true_type, MssDescs>::type;
        using has_extent = typename with_operators<is_esf_with_extent, gt_or>::template iterate_on_esfs<std::false_type,
            MssDescs>::type;

        GRIDTOOLS_STATIC_ASSERT((has_extent::value == has_all_extents::value),
            "The computation appears to have stages with and without extents being specified at the same time. A "
            "computation should have all stages with extents or none.");
        using type = typename boost::mpl::not_<has_all_extents>::type;
    };

    namespace _impl {

        template <int I, class Layout>
        using exists_in_layout = bool_constant < I<Layout::masked_length>;

        template <int I, uint_t Id, class Layout, class Halo, class Alignment>
        enable_if_t<exists_in_layout<I, Layout>::value, bool> storage_info_dim_fits(
            storage_info_interface<Id, Layout, Halo, Alignment> const &storage_info, int val) {
            return val + 1 <= storage_info.template total_length<I>();
        }
        template <int I, uint_t Id, class Layout, class Halo, class Alignment>
        enable_if_t<!exists_in_layout<I, Layout>::value, bool> storage_info_dim_fits(
            storage_info_interface<Id, Layout, Halo, Alignment> const &, int) {
            return true;
        }

        template <class Backend, class Grid>
        struct storage_info_fits_grid_f {
            Grid const &grid;

            template <uint_t Id, class Layout, class Halo, class Alignment>
            bool operator()(storage_info_interface<Id, Layout, Halo, Alignment> const &src) const {

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

    namespace _impl {
        struct dummy_run_f {
            template <typename T>
            reduction_type<T> operator()(T const &) const;
        };
    } // namespace _impl

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
        GRIDTOOLS_STATIC_ASSERT((is_backend<Backend>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

        GRIDTOOLS_STATIC_ASSERT((conjunction<is_condition_tree_of<MssDescriptors, is_computation_token>...>::value),
            "make_computation args should be mss descriptors or condition trees of mss descriptors");

        using branch_selector_t = branch_selector<MssDescriptors...>;
        using all_mss_descriptors_t = typename branch_selector_t::all_leaves_t;
        using return_type = decltype(std::declval<branch_selector_t>().apply(_impl::dummy_run_f{}));

        typedef typename Backend::backend_traits_t::performance_meter_t performance_meter_t;

        using placeholders_t = GT_META_CALL(extract_placeholders, all_mss_descriptors_t);
        using tmp_placeholders_t = GT_META_CALL(meta::filter, (is_tmp_arg, placeholders_t));
        using non_tmp_placeholders_t = GT_META_CALL(meta::filter, (meta::not_<is_tmp_arg>::apply, placeholders_t));

#if GT_BROKEN_TEMPLATE_ALIASES
        template <class Arg>
        struct to_arg_storage_pair {
            using type = arg_storage_pair<Arg, typename Arg::data_store_t>;
        };
        using tmp_arg_storage_pair_tuple_t = GT_META_CALL(meta::rename,
            (meta::defer<std::tuple>::apply, GT_META_CALL(meta::transform, (to_arg_storage_pair, tmp_placeholders_t))));
#else
        template <class Arg>
        using to_arg_storage_pair = arg_storage_pair<Arg, typename Arg::data_store_t>;

        using tmp_arg_storage_pair_tuple_t =
            meta::rename<std::tuple, meta::transform<to_arg_storage_pair, tmp_placeholders_t>>;
#endif

        GRIDTOOLS_STATIC_ASSERT((conjunction<meta::st_contains<non_tmp_placeholders_t, BoundPlaceholders>...>::value),
            "some bound placeholders are not used in mss descriptors");

        GRIDTOOLS_STATIC_ASSERT(
            meta::is_set_fast<meta::list<BoundPlaceholders...>>::value, "bound placeholders should be all different");

        template <class Arg>
        using is_free = negation<meta::st_contains<meta::list<BoundPlaceholders...>, Arg>>;

        using free_placeholders_t = GT_META_CALL(meta::filter, (is_free, non_tmp_placeholders_t));

        using storage_info_map_t = _impl::storage_info_map_t<placeholders_t>;

        using bound_arg_storage_pair_tuple_t =
            std::tuple<_impl::bound_arg_storage_pair<BoundPlaceholders, BoundDataStores>...>;

      public:
        // First we need to compute the association between placeholders and extents.
        // This information is needed to allocate temporaries, and to provide the
        // extent information to the user.
        using extent_map_t = typename boost::mpl::eval_if<typename need_to_compute_extents<all_mss_descriptors_t>::type,
            placeholder_to_extent_map<all_mss_descriptors_t, placeholders_t>,
            boost::mpl::void_>::type;

      private:
        template <typename MssDescs>
        using convert_to_mss_components_array_t =
            copy_into_variadic<typename build_mss_components_array<typename Backend::mss_fuse_esfs_strategy,
                                   MssDescs,
                                   extent_map_t,
                                   typename Grid::axis_type>::type,
                std::tuple<>>;

        using mss_components_array_t = convert_to_mss_components_array_t<all_mss_descriptors_t>;

        using max_extent_for_tmp_t = typename _impl::get_max_extent_for_tmp<mss_components_array_t>::type;

      public:
        // creates a tuple of local domains
        using local_domains_t = GT_META_CALL(_impl::get_local_domains, (mss_components_array_t, IsStateful));

      private:
        struct run_f {
            template <typename MssDescs>
            reduction_type<MssDescs> operator()(
                MssDescs const &mss_descriptors, Grid const &grid, local_domains_t const &local_domains) const {
                auto reduction_data = make_reduction_data(mss_descriptors);
                Backend::template run<convert_to_mss_components_array_t<MssDescs>>(grid, local_domains, reduction_data);
                return reduction_data.reduced_value();
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
              // that ensures, that only
              m_tmp_arg_storage_pair_tuple(dedup_storage_info(
                  _impl::make_tmp_arg_storage_pairs<max_extent_for_tmp_t, Backend, tmp_arg_storage_pair_tuple_t>(
                      grid))),
              // stash bound storages; sanitizing them through the `dedup_storage_info` as well.
              m_bound_arg_storage_pair_tuple(dedup_storage_info(std::move(arg_storage_pairs))) {
            if (timer_enabled)
                m_meter.reset(new performance_meter_t{"NoName"});

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
        template <class... Args, class... DataStores>
        typename std::enable_if<sizeof...(Args) == meta::length<free_placeholders_t>::value, return_type>::type run(
            arg_storage_pair<Args, DataStores> const &... srcs) {
            if (m_meter)
                m_meter->start();
            GRIDTOOLS_STATIC_ASSERT((conjunction<meta::st_contains<free_placeholders_t, Args>...>::value),
                "some placeholders are not used in mss descriptors");
            GRIDTOOLS_STATIC_ASSERT(
                meta::is_set_fast<meta::list<Args...>>::value, "free placeholders should be all different");

            // make views from bound storages again (in the case old views got inconsistent)
            // make views from free storages;
            // concatenate them into a single tuple.
            // push view for updating the local domains.
            update_local_domains(std::tuple_cat(make_view_infos(m_bound_arg_storage_pair_tuple),
                make_view_infos(dedup_storage_info(std::tie(srcs...)))));
            // now local domains are fully set up.
            // branch selector calls run_f functor on the right branch of mss condition tree.
            auto res = m_branch_selector.apply(run_f{}, std::cref(m_grid), std::cref(m_local_domains));
            if (m_meter)
                m_meter->pause();
            return res;
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

        local_domains_t const &local_domains() const { return m_local_domains; }

      private:
        template <class Src>
        static auto make_view_infos(Src &&src)
            GT_AUTO_RETURN(tuple_util::transform(_impl::make_view_info_f<Backend>{}, std::forward<Src>(src)));

        template <class Views>
        void update_local_domains(Views const &views) {
            _impl::update_local_domains(views, m_local_domains);
        }

        template <class Seq>
        auto dedup_storage_info(Seq const &seq) GT_AUTO_RETURN(
            tuple_util::transform(_impl::dedup_storage_info_f<storage_info_map_t>{m_storage_info_map}, seq));
    };

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

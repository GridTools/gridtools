/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>

#include <boost/mpl/and.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/transform.hpp>

#include "../../../common/defs.hpp"
#include "../../../common/generic_metafunctions/for_each.hpp"
#include "../../../common/generic_metafunctions/is_sequence_of.hpp"
#include "../../../meta/macros.hpp"
#include "../../../meta/make_indices.hpp"
#include "../../backend_ids.hpp"
#include "../../mss_components.hpp"
#include "../../mss_functor.hpp"
#include "./execinfo_mc.hpp"

namespace gridtools {

    template <class>
    struct strategy_from_id_mc;

    namespace _impl {

        /**
         * @brief Meta function to check if an MSS can be executed in parallel along k-axis.
         */
        struct is_mss_kparallel {
            template <typename Mss>
            struct apply {
                using type = execute::is_parallel<typename Mss::execution_engine_t>;
            };
        };

        /**
         * @brief Meta function to check if all MSS in an MssComponents array can be executed in parallel along k-axis.
         */
        template <typename MssComponents>
        struct all_mss_kparallel
            : boost::mpl::fold<typename boost::mpl::transform<MssComponents, is_mss_kparallel>::type,
                  boost::mpl::true_,
                  boost::mpl::and_<boost::mpl::placeholders::_1, boost::mpl::placeholders::_2>>::type {};

    } // namespace _impl

    /**
     * @brief Specialization for the \ref gridtools::strategy::block strategy.
     */
    template <>
    struct strategy_from_id_mc<strategy::block> {
        /**
         * @brief Loops over all blocks and executes sequentially all MSS functors for each block.
         * Implementation for stencils with serial execution along k-axis.
         *
         * @tparam MssComponents A meta array with the MSS components of all MSS.
         * @tparam BackendIds IDs of backend.
         */
        template <typename MssComponents, typename BackendIds, typename Enable = void>
        struct fused_mss_loop {
            GT_STATIC_ASSERT((is_sequence_of<MssComponents, is_mss_components>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT((is_backend_ids<BackendIds>::value), GT_INTERNAL_ERROR);

            template <typename LocalDomainListArray, typename Grid>
            GT_FUNCTION static void run(LocalDomainListArray const &local_domain_lists, Grid const &grid) {
                using iter_range = GT_META_CALL(meta::make_indices, boost::mpl::size<MssComponents>);
                using mss_functor_t =
                    mss_functor<MssComponents, Grid, LocalDomainListArray, BackendIds, execinfo_mc::block_kserial_t>;

                execinfo_mc exinfo(grid);
                const int_t i_blocks = exinfo.i_blocks();
                const int_t j_blocks = exinfo.j_blocks();
#pragma omp parallel for collapse(2)
                for (int_t bj = 0; bj < j_blocks; ++bj) {
                    for (int_t bi = 0; bi < i_blocks; ++bi) {
                        gridtools::for_each<iter_range>(mss_functor_t(local_domain_lists, grid, exinfo.block(bi, bj)));
                    }
                }
            }
        };

        /**
         * @brief Loops over all blocks and executes sequentially all MSS functors for each block.
         * Implementation for stencils with parallel execution along k-axis.
         *
         * @tparam MssComponents A meta array with the MSS components of all MSS.
         * @tparam BackendIds IDs of backend.
         */
        template <typename MssComponents, typename BackendIds>
        struct fused_mss_loop<MssComponents,
            BackendIds,
            typename std::enable_if<_impl::all_mss_kparallel<MssComponents>::value>::type> {
            GT_STATIC_ASSERT((is_sequence_of<MssComponents, is_mss_components>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT((is_backend_ids<BackendIds>::value), GT_INTERNAL_ERROR);

            template <typename LocalDomainListArray, typename Grid>
            GT_FUNCTION static void run(LocalDomainListArray const &local_domain_lists, Grid const &grid) {
                using iter_range = GT_META_CALL(meta::make_indices, boost::mpl::size<MssComponents>);
                using mss_functor_t =
                    mss_functor<MssComponents, Grid, LocalDomainListArray, BackendIds, execinfo_mc::block_kparallel_t>;

                execinfo_mc exinfo(grid);
                const int_t i_blocks = exinfo.i_blocks();
                const int_t j_blocks = exinfo.j_blocks();
                const int_t k_first = grid.k_min();
                const int_t k_last = grid.k_max();
#pragma omp parallel for collapse(3)
                for (int_t bj = 0; bj < j_blocks; ++bj) {
                    for (int_t k = k_first; k <= k_last; ++k) {
                        for (int_t bi = 0; bi < i_blocks; ++bi) {
                            gridtools::for_each<iter_range>(
                                mss_functor_t(local_domain_lists, grid, exinfo.block(bi, bj, k)));
                        }
                    }
                }
            }
        };
    };

} // namespace gridtools

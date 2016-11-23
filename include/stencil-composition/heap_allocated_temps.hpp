/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include <boost/fusion/include/filter_view.hpp>
#include <boost/fusion/include/for_each.hpp>

#include "common/defs.hpp"
#include "common/is_temporary_storage.hpp"

#include "storage/base_storage.hpp"

#include "backend_ids.hpp"
#include "stencil-composition/backend_fwd.hpp"
#include "stencil-composition/grid.hpp"
#include "storage/metadata_set.hpp"

/**
@file
TODO Document me!
*/

namespace gridtools {
    namespace _impl {

        /** prepare temporaries struct, constructing the domain for the temporary fields, with the arguments
            to the constructor depending on the specific strategy */
        template < typename ArgList, typename MetaList, typename Grid, typename BackendIds >
        struct prepare_temporaries_functor;

        /**
           Specialization for Naive policy
         */
        template < typename ArgList,
            typename MetaList,
            typename Grid,
            enumtype::platform BackendId,
            enumtype::grid_type GridId >
        struct prepare_temporaries_functor< ArgList,
            MetaList,
            Grid,
            backend_ids< BackendId, GridId, enumtype::Naive > > {

            // TODO check the type of ArgList
            GRIDTOOLS_STATIC_ASSERT(is_metadata_set< MetaList >::value, "wrong type for metadata");
            GRIDTOOLS_STATIC_ASSERT(is_grid< Grid >::value, "wrong type for grid");

            typedef MetaList metadata_set_t;

            /**
               @brief instantiate the \ref gridtools::aggregator_type for the temporary storages
            */
            struct instantiate_tmps {
                uint_t m_tile_i; // tile along i
                uint_t m_tile_j; // tile along j
                uint_t m_tile_k; // tile along k
                metadata_set_t &m_metadata_set;

                GT_FUNCTION
                instantiate_tmps(metadata_set_t &metadata_set_, uint_t tile_i, uint_t tile_j, uint_t tile_k)
                    : m_metadata_set(metadata_set_), m_tile_i(tile_i), m_tile_j(tile_j), m_tile_k(tile_k) {}

                // ElemType: an element in the data field place-holders list
                template < typename ElemType >
                void operator()(pointer< ElemType > &e) const {

                    GRIDTOOLS_STATIC_ASSERT(is_storage< ElemType >::value, "wrong type");
                    GRIDTOOLS_STATIC_ASSERT(ElemType::is_temporary, "wrong type (not temporary)");
                    GRIDTOOLS_STATIC_ASSERT(
                        is_meta_storage< typename ElemType::storage_info_type >::value, "wrong metadata type");

                    typedef typename ElemType::storage_info_type meta_t;

                    // ElemType::info_string.c_str();
                    // calls the constructor of the storage
                    meta_t meta_data(m_tile_i, m_tile_j, m_tile_k);
                    e = new ElemType(meta_data, "default tmp storage", true /*do_allocate*/);

                    // insert new type in the map only if not present already
                    if (!m_metadata_set.template present< pointer< typename ElemType::storage_info_type const > >())
                        // get the meta_data pointer from the temporary storage and insert it into the metadata_set
                        m_metadata_set.insert(e->get_meta_data_pointer());
                }
            };

            static void prepare_temporaries(ArgList &arg_list, metadata_set_t &metadata_, Grid const &grid) {

#ifdef VERBOSE
                std::cout << "Prepare ARGUMENTS" << std::endl;
#endif

                typedef boost::fusion::filter_view< ArgList, is_temporary_storage< boost::mpl::_1 > > view_type;

                view_type fview(arg_list);

                boost::fusion::for_each(fview,
                    instantiate_tmps(metadata_,
                                            grid.direction_i().total_length(),
                                            grid.direction_j().total_length(),
                                            grid.value_at_top() + 1));
            }
        };

        /**
           Specialization for Block policy
         */
        template < typename ArgList,
            typename MetaList,
            typename Grid,
            enumtype::platform BackendId,
            enumtype::grid_type GridId >
        struct prepare_temporaries_functor< ArgList,
            MetaList,
            Grid,
            backend_ids< BackendId, GridId, enumtype::Block > > {

            // TODO implement a check for the ArgList type
            GRIDTOOLS_STATIC_ASSERT(is_metadata_set< MetaList >::value, "wrong type for metadata");
            GRIDTOOLS_STATIC_ASSERT(is_grid< Grid >::value, "wrong type for Grid");

            typedef backend< BackendId, GridId, enumtype::Block > backend_type;
            /**
               @brief instantiate the \ref gridtools::aggregator_type for the temporary storages
            */
            struct instantiate_tmps {
                typedef MetaList metadata_set_t;
                metadata_set_t &m_metadata_set;
                uint_t m_offset_i; // offset along i
                uint_t m_offset_j; // offset along j
                uint_t m_offset_k; // offset along k
                uint_t m_n_i_threads;
                uint_t m_n_j_threads;

                GT_FUNCTION
                instantiate_tmps(metadata_set_t &metadata_set_,
                    uint_t offset_i,
                    uint_t offset_j,
                    uint_t offset_k,
                    uint_t m_n_i_threads,
                    uint_t m_n_j_threads)
                    : m_metadata_set(metadata_set_), m_offset_i(offset_i), m_offset_j(offset_j), m_offset_k(offset_k),
                      m_n_i_threads(m_n_i_threads), m_n_j_threads(m_n_j_threads) {}

                // ElemType: an element in the data field place-holders list
                template < typename ElemType >
                void operator()(pointer< ElemType > &e) const {
                    GRIDTOOLS_STATIC_ASSERT(is_storage< ElemType >::value, "wrong type (not storage)");
                    GRIDTOOLS_STATIC_ASSERT(ElemType::is_temporary, "wrong type (not temporary)");
                    GRIDTOOLS_STATIC_ASSERT(
                        is_meta_storage< typename ElemType::storage_info_type >::value, "wrong metadata type");

                    typedef typename ElemType::storage_info_type meta_t;

                    // calls the constructor of the storage
                    meta_t meta_data(m_offset_i, m_offset_j, m_offset_k, m_n_i_threads, m_n_j_threads);
                    e = new ElemType(meta_data, "blocked tmp storage", true /*do_allocate*/);

                    // insert new type in the map only if not present already
                    if (!m_metadata_set.template present< pointer< const meta_t > >())
                        // get the meta_data pointer from the temporary storage and insert it into the metadata_set
                        m_metadata_set.insert(e->get_meta_data_pointer());
                }
            };

            static void prepare_temporaries(ArgList &arg_list, MetaList &metadata_, Grid const &grid) {
// static const enumtype::strategy StrategyType = Block;

#ifdef VERBOSE
                std::cout << "Prepare ARGUMENTS" << std::endl;
#endif
                typedef boost::fusion::filter_view< ArgList, is_temporary_storage< boost::mpl::_1 > > view_type;

                view_type fview(arg_list);
                boost::fusion::for_each(fview,
                    instantiate_tmps(metadata_,
                                            grid.i_low_bound(),
                                            grid.j_low_bound(),
                                            grid.value_at_top() + 1,
                                            backend_type::n_i_pes()(grid.i_high_bound() - grid.i_low_bound()),
                                            backend_type::n_j_pes()(grid.j_high_bound() - grid.j_low_bound())));
            }
        };

        struct delete_tmps {
            template < typename Elem >
            GT_FUNCTION void operator()(Elem &elem) const {
                delete_pointer d;
                d(elem);
            }
        };

    } // namespace _impl

} // namespace gridtools

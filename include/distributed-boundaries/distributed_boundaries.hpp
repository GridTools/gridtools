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

#include "./binded_bc.hpp"
#include "../common/boollist.hpp"
#include "../common/halo_descriptor.hpp"
#include "../gridtools.hpp"
#include "../stencil-composition/stencil-composition.hpp"
#include "../communication/GCL.hpp"
#include "../communication/low-level/proc_grids_3D.hpp"
#include "../communication/halo_exchange.hpp"
#include "../boundary-conditions/predicate.hpp"

namespace gridtools {
    template < typename CTraits >
    struct distributed_boundaries {

        using pattern_type = halo_exchange_dynamic_ut< typename CTraits::data_layout,
            typename CTraits::proc_layout,
            typename CTraits::value_type,
            typename CTraits::proc_grid_type,
            typename CTraits::comm_arch_type,
            CTraits::version >;

        array< halo_descriptor, 3 > m_halos;
        array< int_t, 3 > m_sizes;
        pattern_type m_he;

        distributed_boundaries(
            array< halo_descriptor, 3 > halos, boollist< 3 > period, uint_t max_stores, MPI_Comm CartComm)
            : m_halos{halos}, m_sizes{0, 0, 0}, m_he(period, CartComm, m_sizes) {}

        template < typename BoundaryApply, typename ArgsTuple, uint_t... Ids >
        void call_apply(BoundaryApply boundary_apply, ArgsTuple args, gt_integer_sequence< uint_t, Ids... >) {
            boundary_apply.apply(std::get< Ids >(args)...);
        }

        template < typename BCApply >
        typename std::enable_if< is_binded_bc< BCApply >::value, void >::type apply_boundary(BCApply bcapply) {
            /*Apply doundaty to data*/
            call_apply(boundary< typename BCApply::boundary_class,
                           CTraits::compute_arch,
                           proc_grid_predicate< typename CTraits::proc_grid_type > >(m_halos,
                           bcapply.boundary_to_apply(),
                           proc_grid_predicate< typename CTraits::proc_grid_type >(m_he.comm())),
                bcapply.stores(),
                typename make_gt_integer_sequence< uint_t,
                           std::tuple_size< typename BCApply::stores_type >::value >::type{});
            std::cout << "Apply job\n";
        }

        template < typename BCApply >
        typename std::enable_if< not is_binded_bc< BCApply >::value, void >::type apply_boundary(BCApply) {
            /* do nothing for a pure data_store*/
            std::cout << "Nothing to apply\n";
        }

        template < typename FirstJob >
        auto collect_stores(
            FirstJob firstjob, typename std::enable_if< is_binded_bc< FirstJob >::value, void * >::type = nullptr)
            -> decltype(firstjob.stores()) {
            return firstjob.stores();
        }

        template < typename FirstJob >
        auto collect_stores(
            FirstJob first_job, typename std::enable_if< not is_binded_bc< FirstJob >::value, void * >::type = nullptr)
            -> decltype(std::make_tuple(first_job)) {
            return std::make_tuple(first_job);
        }

        template < typename Stores, uint_t... Ids >
        void call_pack(Stores const &stores, gt_integer_sequence< uint_t, Ids... >) {
            m_he.pack(advanced::get_initial_address_of(make_host_view(std::get< Ids >(stores)))...);
        }

        template < typename Stores, uint_t... Ids >
        void call_unpack(Stores const &stores, gt_integer_sequence< uint_t, Ids... >) {
            m_he.unpack(advanced::get_initial_address_of(make_host_view(std::get< Ids >(stores)))...);
        }

        template < typename... Jobs >
        void exchange(Jobs... jobs) {
            auto all_stores = std::tuple_cat(collect_stores(jobs)...);
            call_pack(all_stores,
                typename make_gt_integer_sequence< uint_t, std::tuple_size< decltype(all_stores) >::value >::type{});
            m_he.exchange();
            using execute_in_order = int[];
            execute_in_order{(apply_boundary(jobs), 0)...};
            call_unpack(all_stores,
                typename make_gt_integer_sequence< uint_t, std::tuple_size< decltype(all_stores) >::value >::type{});
        }

        typename CTraits::proc_grid_type const &proc_grid() const { return m_he.comm(); }
    };

} // namespace gridtools

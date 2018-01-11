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

#include "./bound_bc.hpp"
#include "../common/boollist.hpp"
#include "../common/halo_descriptor.hpp"
#include "../gridtools.hpp"
#include "../stencil-composition/stencil-composition.hpp"
#include "../communication/GCL.hpp"
#include "../communication/low-level/proc_grids_3D.hpp"
#include "../communication/halo_exchange.hpp"
#include "../boundary-conditions/predicate.hpp"

namespace gridtools {

    namespace _workaround {
        /** \internal Workaround for NVCC that has troubles with tuple_cat */
        template < typename... Tuples >
        struct pairwise_tuple_cat;

        template < typename Tuple >
        struct pairwise_tuple_cat< Tuple > {
            static Tuple apply(Tuple t) { return t; }
        };

        template < typename Tuple1, typename Tuple2, typename... Rest >
        struct pairwise_tuple_cat< Tuple1, Tuple2, Rest... > {
            static auto apply(Tuple1 t, Tuple2 s, Rest... rest) -> decltype(
                pairwise_tuple_cat< decltype(std::tuple_cat(t, s)), Rest... >::apply(std::tuple_cat(t, s), rest...)) {
                return pairwise_tuple_cat< decltype(std::tuple_cat(t, s)), Rest... >::apply(
                    std::tuple_cat(t, s), rest...);
            }
        };

        template < typename... Tuples >
        auto tuple_cat(Tuples... ts) -> decltype(pairwise_tuple_cat< Tuples... >::apply(ts...)) {
            return pairwise_tuple_cat< Tuples... >::apply(ts...);
        };
    } // namespace _workaround

    /**
        @brief This class takes a communication traits class and provide a facility to
        perform boundary conditions and communications in a single call.

        After construction a call to gridtools::distributed_boundaries::exchange takes
        a list of gridtools::data_store or girdtools::bound_bc. The data stores will be
        directly used in communication primitives for performing halo_update operation,
        while bound_bc elements will be priocessed by exracting the data stores that need
        communication and others that will go through boundary condition application as
        specified in the bound_bc class.

        Example of use (where `a`, `b`, `c`, and `d` are of data_store type:
        \verbatim
            using storage_info_t = storage_tr::storage_info_t< 0, 3, halo< 2, 2, 0 > >;
            using storage_type = storage_tr::data_store_t< triplet, storage_info_t >;

            halo_descriptor di{halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
            halo_descriptor dj{halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};
            halo_descriptor dk{0, 0, 0, d3 - 1, d3};
            array< halo_descriptor, 3 > halos{di, dj, dk};

            using cabc_t = distributed_boundaries< comm_traits< storage_type, gcl_cpu > >;

            cabc_t cabc{halos, // halos for communication
                        {false, false, false}, // Periodicity in first, second and third dimension
                        4, // Maximum number of data_stores to be handled by this communicatio obeject
                        GCL_WORLD}; // Communicator to be used

            cabc.exchange(bind_bc(value_boundary< triplet >{triplet{42, 42, 42}}, a),
                          bind_bc(copy_boundary{}, b, _1).associate(c),
                          d);
        \endverbatim

        \tparal CTraits Communication traits. To see an example see gridtools::comm_traits
    */
    template < typename CTraits >
    struct distributed_boundaries {

        using pattern_type = halo_exchange_dynamic_ut< typename CTraits::data_layout,
            typename CTraits::proc_layout,
            typename CTraits::value_type,
            typename CTraits::proc_grid_type,
            typename CTraits::comm_arch_type,
            CTraits::version >;

      private:
        array< halo_descriptor, 3 > m_halos;
        array< int_t, 3 > m_sizes;
        uint_t m_max_stores;
        pattern_type m_he;

      public:
        /**
            @brief Constructor of distributed_boundaries.

            \param gridtools::halos array of 3 gridtools::halo_desctiptor containing the halo information to be used for
           communication
            \param period Periodicity specification, a gridtools::boollist with three elements, one for each dimension.
           true mean the dimension is periodic
            \param max_stores Maximum number of data_stores to be used in communication. PAssing more will couse a
           runtime error (probably segmentation fault), passing less will underutilize the memory
            \param MPI_Comm MPI communicator to use in the halo update operation.
        */
        distributed_boundaries(
            array< halo_descriptor, 3 > halos, boollist< 3 > period, uint_t max_stores, MPI_Comm CartComm)
            : m_halos{halos}, m_sizes{0, 0, 0}, m_max_stores{max_stores}, m_he(period, CartComm, m_sizes) {
            m_he.template add_halo< 0 >(
                m_halos[0].minus(), m_halos[0].plus(), m_halos[0].begin(), m_halos[0].end(), m_halos[0].total_length());

            m_he.template add_halo< 1 >(
                m_halos[1].minus(), m_halos[1].plus(), m_halos[1].begin(), m_halos[1].end(), m_halos[1].total_length());

            m_he.template add_halo< 2 >(
                m_halos[2].minus(), m_halos[2].plus(), m_halos[2].begin(), m_halos[2].end(), m_halos[2].total_length());

            m_he.setup(m_max_stores);
        }

        /**
            @brief Member function to perform boundary condition and communication on a list of jobs.
            A job is either a gridtools::data_store to be used during communication or a gridtools::bound_bc
            to apply boundary conditions and halo_update operations for the data_stores that are not input-only
            (that will be indicated with the gridtools::bound_bc::associate member function.)

            \param jobs Variadic list of jobs
        */
        template < typename... Jobs >
        void exchange(Jobs const &... jobs) {
#ifdef __CUDACC__
            // Workaround for cuda to handle tuple_cat. Compilation is a little slower.
            // This can be removed when nvcc supports it.
            auto all_stores_for_exc = _workaround::tuple_cat(collect_stores(jobs)...);
#else
            auto all_stores_for_exc = std::tuple_cat(collect_stores(jobs)...);
#endif
            if (m_max_stores < std::tuple_size< decltype(all_stores_for_exc) >::value) {
                std::string err{"Too many data stores to be exchanged" +
                                std::to_string(std::tuple_size< decltype(all_stores_for_exc) >::value) +
                                " instead of the maximum allowed, which is " + std::to_string(m_max_stores)};
                throw std::runtime_error(err);
            }

            using execute_in_order = int[];
            execute_in_order{(apply_boundary(jobs), 0)...};
            call_pack(all_stores_for_exc,
                typename make_gt_integer_sequence< uint_t,
                          std::tuple_size< decltype(all_stores_for_exc) >::value >::type{});
            m_he.exchange();
            call_unpack(all_stores_for_exc,
                typename make_gt_integer_sequence< uint_t,
                            std::tuple_size< decltype(all_stores_for_exc) >::value >::type{});
        }

        typename CTraits::proc_grid_type const &proc_grid() const { return m_he.comm(); }

      private:
        template < typename BoundaryApply, typename ArgsTuple, uint_t... Ids >
        void call_apply(BoundaryApply boundary_apply, ArgsTuple const &args, gt_integer_sequence< uint_t, Ids... >) {
            boundary_apply.apply(std::get< Ids >(args)...);
        }

        template < typename BCApply >
        typename std::enable_if< is_bound_bc< BCApply >::value, void >::type apply_boundary(BCApply bcapply) {
            /*Apply boundary to data*/
            call_apply(boundary< typename BCApply::boundary_class,
                           CTraits::compute_arch,
                           proc_grid_predicate< typename CTraits::proc_grid_type > >(m_halos,
                           bcapply.boundary_to_apply(),
                           proc_grid_predicate< typename CTraits::proc_grid_type >(m_he.comm())),
                bcapply.stores(),
                typename make_gt_integer_sequence< uint_t,
                           std::tuple_size< typename BCApply::stores_type >::value >::type{});
        }

        template < typename BCApply >
        typename std::enable_if< not is_bound_bc< BCApply >::value, void >::type apply_boundary(BCApply) {
            /* do nothing for a pure data_store*/
        }

        template < typename FirstJob >
        auto collect_stores(
            FirstJob const &firstjob, typename std::enable_if< is_bound_bc< FirstJob >::value, void * >::type = nullptr)
            -> decltype(firstjob.exc_stores()) {
            return firstjob.exc_stores();
        }

        template < typename FirstJob >
        auto collect_stores(FirstJob const &first_job,
            typename std::enable_if< not is_bound_bc< FirstJob >::value, void * >::type = nullptr)
            -> decltype(std::make_tuple(first_job)) {
            return std::make_tuple(std::cref(first_job));
        }

        template < typename Stores, uint_t... Ids >
        void call_pack(Stores const &stores, gt_integer_sequence< uint_t, Ids... >) {
            m_he.pack(advanced::get_initial_address_of(_impl::proper_view< CTraits::compute_arch,
                access_mode::ReadWrite,
                typename std::decay< typename std::tuple_element< Ids, Stores >::type >::type >::
                    make(std::get< Ids >(stores)))...);
        }

        template < typename Stores, uint_t... Ids >
        void call_pack(Stores const &stores, gt_integer_sequence< uint_t >) {}

        template < typename Stores, uint_t... Ids >
        void call_unpack(Stores const &stores, gt_integer_sequence< uint_t, Ids... >) {
            m_he.unpack(advanced::get_initial_address_of(_impl::proper_view< CTraits::compute_arch,
                access_mode::ReadWrite,
                typename std::decay< typename std::tuple_element< Ids, Stores >::type >::type >::
                    make(std::get< Ids >(stores)))...);
        }
    };

} // namespace gridtools

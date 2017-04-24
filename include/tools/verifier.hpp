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

#include "common/array.hpp"
#include "common/array_addons.hpp"
#include "common/gt_math.hpp"

namespace gridtools {

    template < typename Partitioner, typename Storage >
    class parallel_storage_info;

    template < typename value_type >
    GT_FUNCTION bool compare_below_threshold(value_type expected, value_type actual, double precision) {
        value_type M = math::max(math::fabs(expected), math::fabs(actual));
        value_type m = math::min(math::fabs(expected), math::fabs(actual));
        value_type e = (M - m) / M;
        if ((m == M) || (e <= precision) || ((M - m) < precision)) {
            return true;
        }
        return false;
    }

#ifdef CXX11_ENABLED

    template < typename Array, typename StorageInfo, typename... T >
    typename boost::enable_if_c< (Array::n_dimensions == sizeof...(T)), const int >::type get_index(
        Array const &pos, StorageInfo storage_info, T... t) {
        return storage_info.index(t...);
    }

    template < typename Array, typename StorageInfo, typename... T >
    typename boost::enable_if_c< (Array::n_dimensions > sizeof...(T)), const int >::type get_index(
        Array const &pos, StorageInfo storage_info, T... t) {
        return get_index(pos, storage_info, t..., pos[sizeof...(T)]);
    }

    template < uint_t NDim, uint_t NCoord, typename StorageType >
    struct verify_helper {
        verify_helper(StorageType const &exp_field,
            StorageType const &actual_field,
            uint_t field_id,
            array< array< uint_t, 2 >, StorageType::storage_info_t::layout_t::length > const &halos,
            double precision)
            : m_exp_field(exp_field), m_actual_field(actual_field), m_field_id(field_id), m_precision(precision),
              m_halos(halos) {}

        template < typename Grid >
        bool operator()(Grid const &grid_, array< uint_t, NCoord > const &pos) {
            typename StorageType::storage_info_t const &meta = *(m_exp_field.get_storage_info_ptr());

            const gridtools::uint_t size = meta.template unaligned_dim< NDim - 1 >();
            bool verified = true;

            verify_helper< NDim - 1, NCoord + 1, StorageType > next_loop(
                m_exp_field, m_actual_field, m_field_id, m_halos, m_precision);

            const uint_t halo_minus = m_halos[NDim - 1][0];
            const uint_t halo_plus = m_halos[NDim - 1][1];

            for (int c = halo_minus; c < size - halo_plus; ++c) {
                array< uint_t, NCoord + 1 > new_pos = pos.prepend_dim(c);
                verified = verified & next_loop(grid_, new_pos);
            }
            return verified;
        }

      private:
        StorageType const &m_exp_field;
        StorageType const &m_actual_field;
        double m_precision;
        uint_t m_field_id;
        array< array< uint_t, 2 >, StorageType::storage_info_t::layout_t::length > const &m_halos;
    };

    template < uint_t NCoord, typename StorageType >
    struct verify_helper< 0, NCoord, StorageType > {
        verify_helper(StorageType const &exp_field,
            StorageType const &actual_field,
            uint_t field_id,
            array< array< uint_t, 2 >, StorageType::storage_info_t::layout_t::length > const &halos,
            double precision)
            : m_exp_field(exp_field), m_actual_field(actual_field), m_field_id(field_id), m_precision(precision),
              m_halos(halos) {}

        template < typename Grid >
        bool operator()(Grid const &grid_, array< uint_t, NCoord > const &pos) {
            bool verified = true;
            if (pos[2] < grid_.k_max()) {
                typename StorageType::storage_info_t const &meta = *(m_exp_field.get_storage_info_ptr());

                typename StorageType::data_t expected =
                    m_exp_field.get_storage_ptr()->get_cpu_ptr()[get_index(pos, meta)];
                typename StorageType::data_t actual =
                    m_actual_field.get_storage_ptr()->get_cpu_ptr()[get_index(pos, meta)];
                if (!compare_below_threshold(expected, actual, m_precision)) {

                    std::cout << "Error in field dimension " << m_field_id << " and position " << pos
                              << " ; expected : " << expected << " ; actual : " << actual << "  "
                              << std::fabs((expected - actual) / (expected)) << std::endl;
                    verified = false;
                }
            }
            return verified;
        }

      private:
        StorageType const &m_exp_field;
        StorageType const &m_actual_field;
        uint_t m_field_id;
        double m_precision;
        array< array< uint_t, 2 >, StorageType::storage_info_t::layout_t::length > const &m_halos;
    };

    template < uint_t NDim, typename Grid, typename StorageType >
    bool verify_functor(Grid const &grid_,
        StorageType const &exp_field,
        StorageType const &actual_field,
        uint_t field_id,
        array< array< uint_t, 2 >, StorageType::storage_info_t::layout_t::length > halos,
        double precision) {
        typename StorageType::storage_info_t const &meta = *(exp_field.get_storage_info_ptr());

        const gridtools::uint_t size = meta.template unaligned_dim< NDim - 1 >();
        bool verified = true;
        verify_helper< NDim - 1, 1, StorageType > next_loop(exp_field, actual_field, field_id, halos, precision);

        const uint_t halo_minus = halos[NDim - 1][0];
        const uint_t halo_plus = halos[NDim - 1][1];

        for (uint_t c = halo_minus; c < size - halo_plus; ++c) {
#ifdef CXX11_ENABLED
            array< uint_t, 1 > new_pos{c};
#else
            array< uint_t, 1 > new_pos(c);
#endif
            verified = verified & next_loop(grid_, new_pos);
        }
        return verified;
    }

    class verifier {
      public:
        verifier(const double precision) : m_precision(precision) {}
        ~verifier() {}

        template < typename Grid, typename StorageType >
        bool verify(Grid const &grid_,
            StorageType const &field1,
            StorageType const &field2,
            const array< array< uint_t, 2 >, StorageType::storage_info_t::layout_t::length > halos) {

            bool verified = true;

            for (gridtools::uint_t f = 0; f < 1; ++f) {
                verified = verify_functor< StorageType::storage_info_t::layout_t::length >(
                    grid_, field1, field2, f, halos, m_precision);
            }
            return verified;
        }

      private:
        double m_precision;
    };
#else
    class verifier {
      public:
        verifier(const double precision, const int halo_size) : m_precision(precision), m_halo_size(halo_size) {}
        ~verifier() {}

        template < typename Grid, typename storage_type >
        bool verify(Grid const &grid_, storage_type const &field1, storage_type const &field2) const {
            // assert(field1.template dim<0>() == field2.template dim<0>());
            // assert(field1.template dim<1>() == field2.template dim<1>());
            // assert(field1.template dim<2>() == field2.template dim<2>());
            typename storage_type::storage_info_t const *meta = field1.get_storage_info_ptr();

            const gridtools::uint_t idim = meta->template unaligned_dim< 0 >();
            const gridtools::uint_t jdim = meta->template unaligned_dim< 1 >();
            const gridtools::uint_t kdim = meta->template unaligned_dim< 2 >();

            bool verified = true;

            for (gridtools::uint_t f = 0; f < storage_type::field_dimensions; ++f)
                for (gridtools::uint_t i = m_halo_size; i < idim - m_halo_size; ++i) {
                    for (gridtools::uint_t j = m_halo_size; j < jdim - m_halo_size; ++j) {
                        for (gridtools::uint_t k = 0; k < grid_.k_max(); ++k) {
                            typename storage_type::data_t expected = field1.fields()[f][meta->index(i, j, k)];
                            typename storage_type::data_t actual = field2.fields()[f][meta->index(i, j, k)];

                            if (!compare_below_threashold(expected, actual)) {
                                std::cout << "Error in position " << i << " " << j << " " << k
                                          << " ; expected : " << expected << " ; actual : " << actual << "  "
                                          << std::fabs((expected - actual) / (expected)) << std::endl;
                                verified = false;
                            }
                        }
                    }
                }

            return verified;
        }

        template < typename Grid, typename Partitioner, typename MetaStorageType, typename StorageType >
        bool verify_parallel(Grid const &grid_,
            gridtools::parallel_storage_info< MetaStorageType, Partitioner > const &metadata_,
            StorageType const &field1,
            StorageType const &field2) {

            const gridtools::uint_t idim = metadata_.get_metadata().template unaligned_dim< 0 >();
            const gridtools::uint_t jdim = metadata_.get_metadata().template unaligned_dim< 1 >();
            const gridtools::uint_t kdim = metadata_.get_metadata().template unaligned_dim< 2 >();

            bool verified = true;

            for (gridtools::uint_t f = 0; f < StorageType::field_dimensions; ++f)
                for (gridtools::uint_t i = m_halo_size; i < idim - m_halo_size; ++i) {
                    for (gridtools::uint_t j = m_halo_size; j < jdim - m_halo_size; ++j) {
                        for (gridtools::uint_t k = 0; k < grid_.k_max(); ++k) {
                            if (metadata_.mine(i, j, k)) {
                                typename StorageType::data_t expected = field2.get_value(i, j, k);
                                typename StorageType::data_t actual = field1[metadata_.get_local_index(i, j, k)];

                                if (!compare_below_threashold(expected, actual)) {
                                    std::cout << "Error in position " << i << " " << j << " " << k
                                              << " ; expected : " << expected << " ; actual : " << actual << "  "
                                              << std::fabs((expected - actual) / (expected)) << std::endl;
                                    verified = false;
                                }
                            }
                        }
                    }
                }

            return verified;
        }

      private:
        template < typename value_type >
        bool compare_below_threashold(value_type expected, value_type actual) const {
            if (std::fabs(expected) < 1e-3 && std::fabs(actual) < 1e-3) {
                if (std::fabs(expected - actual) < m_precision)
                    return true;
            } else {
                if (std::fabs((expected - actual) / (m_precision * expected)) < 1.0)
                    return true;
            }
            return false;
        }
        double m_precision;
        int m_halo_size;
    };
#endif

} // namespace gridtools

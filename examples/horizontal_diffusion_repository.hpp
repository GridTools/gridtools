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

#include <gridtools.hpp>
#include <storage/storage-facility.hpp>

namespace horizontal_diffusion {

#ifdef CUDA_EXAMPLE
    typedef gridtools::backend< gridtools::enumtype::Cuda,
        gridtools::enumtype::GRIDBACKEND,
        gridtools::enumtype::Block >
        hd_backend;
    typedef gridtools::storage_traits< gridtools::enumtype::Cuda > storage_tr;
#else
#ifdef BACKEND_BLOCK
    typedef gridtools::backend< gridtools::enumtype::Host,
        gridtools::enumtype::GRIDBACKEND,
        gridtools::enumtype::Block >
        hd_backend;
#else
    typedef gridtools::backend< gridtools::enumtype::Host,
        gridtools::enumtype::GRIDBACKEND,
        gridtools::enumtype::Naive >
        hd_backend;
#endif
    typedef gridtools::storage_traits< gridtools::enumtype::Host > storage_tr;
#endif

    using gridtools::uint_t;
    using gridtools::int_t;

#ifdef __CUDACC__
    typedef gridtools::layout_map< 2, 1, 0 > layout_ijk; // stride 1 on i
    typedef gridtools::layout_map< 1, 0, -1 > layout_ij;
    typedef gridtools::layout_map< -1, 0, -1 > layout_j;
#else
    typedef gridtools::layout_map< 0, 1, 2 > layout_ijk; // stride 1 on k
    typedef gridtools::layout_map< 0, 1, -1 > layout_ij;
    typedef gridtools::layout_map< -1, 0, -1 > layout_j;
#endif

    typedef gridtools::layout_map< -1, -1, -1 > layout_scalar;
#ifdef CXX11_ENABLED
    using storage_info_ijk_t = storage_tr::meta_storage_type< 0, layout_ijk, gridtools::halo< 2, 0, 0 > >;
    using storage_info_ij_t = storage_tr::meta_storage_type< 1, layout_ij, gridtools::halo< 2, 0, 0 > >;
    using storage_info_j_t = storage_tr::meta_storage_type< 2, layout_j, gridtools::halo< 2, 0, 0 > >;
    using storage_info_scalar_t = storage_tr::meta_storage_type< 3, layout_scalar, gridtools::halo< 2, 0, 0 > >;
#else
    typedef storage_tr::meta_storage_type< 0, layout_ijk, gridtools::halo< 2, 0, 0 > >::type storage_info_ijk_t;
    typedef storage_tr::meta_storage_type< 1, layout_ij, gridtools::halo< 2, 0, 0 > >::type storage_info_ij_t;
    typedef storage_tr::meta_storage_type< 2, layout_j, gridtools::halo< 2, 0, 0 > >::type storage_info_j_t;
    typedef storage_tr::meta_storage_type< 3, layout_scalar, gridtools::halo< 2, 0, 0 > >::type storage_info_scalar_t;
#endif
    class repository {
      public:
#ifdef CXX11_ENABLED
        using storage_type = storage_tr::storage_type< gridtools::float_type, storage_info_ijk_t >;
        using ij_storage_type = storage_tr::storage_type< gridtools::float_type, storage_info_ij_t >;
        using j_storage_type = storage_tr::storage_type< gridtools::float_type, storage_info_j_t >;

        using scalar_storage_type = storage_tr::temporary_storage_type< gridtools::float_type, storage_info_scalar_t >;
        using tmp_storage_type = storage_tr::temporary_storage_type< gridtools::float_type, storage_info_ijk_t >;
        using tmp_scalar_storage_type =
            storage_tr::temporary_storage_type< gridtools::float_type, storage_info_scalar_t >;
#else
        typedef storage_tr::storage_type< gridtools::float_type, storage_info_ijk_t >::type storage_type;
        typedef storage_tr::storage_type< gridtools::float_type, storage_info_ij_t >::type ij_storage_type;
        typedef storage_tr::storage_type< gridtools::float_type, storage_info_j_t >::type j_storage_type;

        typedef storage_tr::temporary_storage_type< gridtools::float_type, storage_info_scalar_t >::type
            scalar_storage_type;
        typedef storage_tr::temporary_storage_type< gridtools::float_type, storage_info_ijk_t >::type tmp_storage_type;
        typedef storage_tr::temporary_storage_type< gridtools::float_type, storage_info_scalar_t >::type
            tmp_scalar_storage_type;
#endif

        storage_info_ijk_t m_storage_info_ijk;
        storage_info_j_t m_storage_info_j;

      private:
        storage_info_ij_t m_storage_info_ij;
        storage_info_scalar_t m_storage_info_scalar;
        storage_type in_, out_, out_ref_, coeff_;
        j_storage_type crlato_, crlatu_, crlat0_, crlat1_;
        const uint_t halo_size_;
        const uint_t idim_, jdim_, kdim_;

      public:
        repository(const uint_t idim, const uint_t jdim, const uint_t kdim, const uint_t halo_size)
            : m_storage_info_ijk(idim, jdim, kdim), m_storage_info_ij(idim, jdim, kdim),
              m_storage_info_j(idim, jdim, kdim), m_storage_info_scalar(idim, jdim, kdim),
              in_(m_storage_info_ijk, -1., "in"), crlato_(m_storage_info_j, -1, "crlato"),
              crlatu_(m_storage_info_j, -1, "crlatu"), crlat0_(m_storage_info_j, -1, "crlat0"),
              crlat1_(m_storage_info_j, -1, "crlat1"), out_(m_storage_info_ijk, -1., "out"),
              out_ref_(m_storage_info_ijk, -1., "out_ref"), coeff_(m_storage_info_ijk, -1., "coeff"),
              halo_size_(halo_size), idim_(idim), jdim_(jdim), kdim_(kdim) {}

        void init_fields() {
            const double PI = std::atan(1.) * 4.;
            init_field_to_value(out_, 0.0);
            init_field_to_value(out_ref_, 0.0);

            const uint_t i_begin = 0;
            const uint_t i_end = idim_;
            const uint_t j_begin = 0;
            const uint_t j_end = jdim_;
            const uint_t k_begin = 0;
            const uint_t k_end = kdim_;

            double dx = 1. / (double)(i_end - i_begin);
            double dy = 1. / (double)(j_end - j_begin);

            double delta0 = (0.995156 - 0.994954) / (double)(jdim_ - 1);
            double delta1 = (0.995143 - 0.994924) / (double)(jdim_ - 1);

            for (uint_t j = j_begin; j < j_end; j++) {
                crlat0_(0, j, 0) = 0.994954 + (double)(j)*delta0;
                crlat1_(0, j, 0) = 0.994924 + (double)(j)*delta1;

                for (uint_t i = i_begin; i < i_end; i++) {
                    double x = dx * (double)(i - i_begin);
                    double y = dy * (double)(j - j_begin);
                    for (uint_t k = k_begin; k < k_end; k++) {
                        // in values between 5 and 9
                        in_(i, j, k) = 5. + 8 * (2. + cos(PI * (x + 1.5 * y)) + sin(2 * PI * (x + 1.5 * y))) / 4.;

                        // coefficient values
                        coeff_(i, j, k) = 0.025;
                    }
                }
            }
            for (uint_t j = j_begin + 1; j < j_end; j++) {
                crlato_(0, j, 0) = crlat1_(0, j, 0) / crlat0_(0, j, 0);
                crlatu_(0, j, 0) = crlat1_(0, j - 1, 0) / crlat0_(0, j, 0);
            }
            crlato_(0, j_begin, 0) = crlat1_(0, j_begin, 0) / crlat0_(0, j_begin, 0);
        }

        template < typename TStorage_type, typename TValue_type >
        void init_field_to_value(TStorage_type &field, TValue_type value) {

            const bool has_dim0 = TStorage_type::basic_type::layout::template at< 0 >() != -1;
            const bool has_dim1 = TStorage_type::basic_type::layout::template at< 1 >() != -1;
            const bool has_dim2 = TStorage_type::basic_type::layout::template at< 2 >() != -1;
            for (uint_t k = 0; k < (has_dim2 ? kdim_ : 1); ++k) {
                for (uint_t i = 0; i < (has_dim0 ? idim_ : 1); ++i) {
                    for (uint_t j = 0; j < (has_dim1 ? jdim_ : 1); ++j) {
                        field(i, j, k) = value;
                    }
                }
            }
        }

        void generate_reference() {
            ij_storage_type lap(m_storage_info_ij, -1., "lap");
            ij_storage_type flx(m_storage_info_ij, -1., "flx");
            ij_storage_type fly(m_storage_info_ij, -1., "fly");

            init_field_to_value(lap, 0.0);

            // kbody
            for (uint_t k = 0; k < kdim_; ++k) {
                for (uint_t i = halo_size_ - 1; i < idim_ - halo_size_ + 1; ++i) {
                    for (uint_t j = halo_size_ - 1; j < jdim_ - halo_size_ + 1; ++j) {
                        lap(i, j, (uint_t)0) =
                            (gridtools::float_type)4 * in_(i, j, k) -
                            (in_(i + 1, j, k) + in_(i, j + 1, k) + in_(i - 1, j, k) + in_(i, j - 1, k));
                    }
                }
                for (uint_t i = halo_size_ - 1; i < idim_ - halo_size_; ++i) {
                    for (uint_t j = halo_size_; j < jdim_ - halo_size_; ++j) {
                        flx(i, j, (uint_t)0) = lap(i + 1, j, (uint_t)0) - lap(i, j, (uint_t)0);
                        if (flx(i, j, (uint_t)0) * (in_(i + 1, j, k) - in_(i, j, k)) > 0)
                            flx(i, j, (uint_t)0) = 0.;
                    }
                }
                for (uint_t i = halo_size_; i < idim_ - halo_size_; ++i) {
                    for (uint_t j = halo_size_ - 1; j < jdim_ - halo_size_; ++j) {
                        fly(i, j, (uint_t)0) = lap(i, j + 1, (uint_t)0) - lap(i, j, (uint_t)0);
                        if (fly(i, j, (uint_t)0) * (in_(i, j + 1, k) - in_(i, j, k)) > 0)
                            fly(i, j, (uint_t)0) = 0.;
                    }
                }
                for (uint_t i = halo_size_; i < idim_ - halo_size_; ++i) {
                    for (uint_t j = halo_size_; j < jdim_ - halo_size_; ++j) {
                        out_ref_(i, j, k) = in_(i, j, k) -
                                            coeff_(i, j, k) * (flx(i, j, (uint_t)0) - flx(i - 1, j, (uint_t)0) +
                                                                  fly(i, j, (uint_t)0) - fly(i, j - 1, (uint_t)0));
                    }
                }
            }
        }

        void generate_reference_simple() {
            ij_storage_type lap(m_storage_info_ij, -1., "lap");
            init_field_to_value(lap, 0.0);

            // kbody
            for (uint_t k = 0; k < kdim_; ++k) {
                for (uint_t i = halo_size_ - 1; i < idim_ - halo_size_ + 1; ++i) {
                    for (uint_t j = halo_size_ - 1; j < jdim_ - halo_size_ + 1; ++j) {
                        lap(i, j, (uint_t)0) = in_(i + 1, j, k) + in_(i - 1, j, k) -
                                               (gridtools::float_type)2 * in_(i, j, k) +
                                               crlato_(i, j, k) * (in_(i, j + 1, k) - in_(i, j, k)) +
                                               crlatu_(i, j, k) * (in_(i, j - 1, k) - in_(i, j, k));
                    }
                }
                for (uint_t i = halo_size_; i < idim_ - halo_size_; ++i) {
                    for (uint_t j = halo_size_; j < jdim_ - halo_size_; ++j) {
                        gridtools::float_type fluxx = lap(i + 1, j, k) - lap(i, j, k);
                        gridtools::float_type fluxx_m = lap(i, j, k) - lap(i - 1, j, k);

                        gridtools::float_type fluxy = crlato_(i, j, k) * (lap(i, j + 1, k) - lap(i, j, k));
                        gridtools::float_type fluxy_m = crlato_(i, j, k) * (lap(i, j, k) - lap(i, j - 1, k));

                        out_ref_(i, j, k) = in_(i, j, k) + ((fluxx_m - fluxx) + (fluxy_m - fluxy)) * coeff_(i, j, k);
                    }
                }
            }
        }

        void update_cpu() {
#ifdef CUDA_EXAMPLE
            out_.d2h_update();
#endif
        }

        storage_type &in() { return in_; }
        storage_type &out() { return out_; }
        storage_type &out_ref() { return out_ref_; }
        storage_type &coeff() { return coeff_; }
        j_storage_type &crlato() { return crlato_; }
        j_storage_type &crlatu() { return crlatu_; }

        storage_info_ijk_t &storage_info_ijk() { return m_storage_info_ijk; }
    };
}

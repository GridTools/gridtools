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
#include <storage/storage-facility.hpp>
#include <stencil-composition/arg.hpp>
#include <stencil-composition/aggregator_type.hpp>
#include <common/defs.hpp>
#include <stencil-composition/stencil.hpp>

namespace domain_reassign {

#ifdef __CUDACC__
#define BACKEND backend< gridtools::enumtype::Cuda, gridtools::enumtype::GRIDBACKEND, gridtools::enumtype::Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< gridtools::enumtype::Host, gridtools::enumtype::GRIDBACKEND, gridtools::enumtype::Block >
#else
#define BACKEND backend< gridtools::enumtype::Host, gridtools::enumtype::GRIDBACKEND, gridtools::enumtype::Naive >
#endif
#endif

    typedef gridtools::storage_traits< gridtools::BACKEND::s_backend_id >::storage_info_t< 0, 3 > storage_info_t;
    typedef gridtools::storage_traits< gridtools::BACKEND::s_backend_id >::data_store_t< gridtools::float_type,
        storage_info_t > storage_t;

    class gt_example {

        typedef gridtools::arg< 0, storage_t > p_in;
        typedef gridtools::arg< 1, storage_t > p_out;
        typedef gridtools::tmp_arg< 2, storage_t > p_tmp;

        typedef boost::mpl::vector< p_in, p_out, p_tmp > accessor_list;

      public:
        gt_example(gridtools::uint_t d1, gridtools::uint_t d2, gridtools::uint_t d3, storage_t& in, storage_t& out);

        void run(storage_t& in, storage_t& out);

        void run_plch(storage_t& in, storage_t& out);

        void run_on(storage_t& in, storage_t& out);

        void run_on_plch(storage_t& in, storage_t& out);

        void finalize();
      private:
        std::shared_ptr< gridtools::computation< gridtools::aggregator_type< accessor_list >, gridtools::notype > >
            m_stencil;
    };
}

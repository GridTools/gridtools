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

#include <interface/logging.h>
#include <storage/storage-facility.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include <interface/wrapper/simple_wrapper.hpp>
#include <utility>

#ifdef __CUDACC__
#define BACKEND_ARCH Cuda
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#define BACKEND_ARCH Host
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

#ifdef USE_TYPE_FLOAT
#define DATA_TYPE float
#elif USE_TYPE_DOUBLE
#define DATA_TYPE double
#elif USE_TYPE_INT
#define DATA_TYPE int
#else
#error "datatype not defined"
#endif

using namespace gridtools;
using namespace enumtype;

using storage_info_t = gridtools::storage_traits< BACKEND_ARCH >::storage_info_t< 0, 3 >;
using data_store_t = gridtools::storage_traits< BACKEND_ARCH >::data_store_t< DATA_TYPE, storage_info_t >;

struct copy_functor {

    typedef accessor< 0, enumtype::in, extent<>, 3 > in;
    typedef accessor< 1, enumtype::inout, extent<>, 3 > out;
    typedef boost::mpl::vector< in, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        eval(out()) = eval(in());
    }
};

class my_simple_wrapper : public simple_wrapper< data_store_t > {
  public:
    using super = simple_wrapper< data_store_t >;
    my_simple_wrapper(std::vector< uint_t > sizes) : super(sizes) {
        Logging::enable();
        fields["out"] = data_store_t("out");
        fields["in"] = data_store_t("in");

        auto grid = make_grid(sizes[0], sizes[1], sizes[2]);
        aggregator_t domain(fields["out"], fields["in"]);

        stencil_ = gridtools::make_computation< gridtools::BACKEND >(domain,
            grid,
            gridtools::make_multistage(execute< forward >(), gridtools::make_stage< copy_functor >(p_in(), p_out())));
        stencil_->ready();
    }

    virtual ~my_simple_wrapper() = default;

    void run() override;

  private:
    using p_out = arg< 0, data_store_t >;
    using p_in = arg< 1, data_store_t >;
    using aggregator_t = gridtools::aggregator_type< boost::mpl::vector< p_out, p_in > >;
    std::shared_ptr< gridtools::computation< aggregator_t, gridtools::notype > > stencil_;
};

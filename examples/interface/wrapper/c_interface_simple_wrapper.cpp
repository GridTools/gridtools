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
#include "c_interface_simple_wrapper.h"

#include <interface/logging.h>
#include "c_bindings/export.hpp"

void my_simple_wrapper::run() {
    LOG_BEGIN("my_simple_wrapper::run()");
    data_store_t &testfield = fields["in"];
    LOG(info) << "fieldname: " << testfield.name();
    testfield.sync(); // you should not do this in a real application
    auto view = make_host_view(testfield);
    LOG(info) << testfield.name() << "(0,1,2)=" << view(0, 1, 2);
    testfield.sync();

    LOG(info) << "stencil_->reassign()";
    stencil_->reassign(p_in() = fields["in"], p_out() = fields["out"]);
    LOG(info) << "stencil_->steady()";
    stencil_->steady();
    LOG(info) << "stencil_->run()";
    stencil_->run();
    LOG_END();
}

std::shared_ptr< wrappable > alloc_simple_wrapper_impl(int ndims, int *dims) {
    std::vector< gridtools::uint_t > sizes(ndims);
    for (size_t i = 0; i < ndims; ++i) {
        sizes[i] = dims[i];
    }
    return std::shared_ptr< wrappable >(new my_simple_wrapper(sizes));
}
GT_EXPORT_BINDING_2(alloc_simple_wrapper, alloc_simple_wrapper_impl);

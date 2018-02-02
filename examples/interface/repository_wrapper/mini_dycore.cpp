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
#include "mini_dycore.h"

#include <interface/logging.h>
#include <functional>
#include "c_bindings/export.hpp"

void wrapped_dycore_repository::run() {
    // dummy: will probably remoed in a next release
}

std::shared_ptr< wrappable > alloc_wrapped_dycore_repository_impl(int ndims, int *dims) {
    std::vector< gridtools::uint_t > sizes(ndims);
    for (size_t i = 0; i < ndims; ++i) {
        sizes[i] = dims[i];
    }
    return std::shared_ptr< wrappable >(new wrapped_dycore_repository(sizes));
}

GT_EXPORT_BINDING_2(alloc_wrapped_dycore_repository, alloc_wrapped_dycore_repository_impl);

// TODO this is a hack as boost::any doesn't deal with conversion between base and derived class
std::shared_ptr< wrapped_dycore_repository > convert_dycore_repo_impl(std::shared_ptr< wrappable > repo_as_wrappable) {
    return std::shared_ptr< wrapped_dycore_repository >(
        dynamic_cast< wrapped_dycore_repository * >(repo_as_wrappable.get()));
}
GT_EXPORT_BINDING_1(convert_dycore_repo, convert_dycore_repo_impl);

mini_dycore::mini_dycore(std::vector< uint_t > sizes, std::shared_ptr< wrapped_dycore_repository > repo)
    : repository_(repo) {
    Logging::enable();

    auto grid = make_grid(sizes[0], sizes[1], sizes[2]);
    aggregator_t domain(repo->out(), repo->in());

    stencil_ = gridtools::make_computation< gridtools::BACKEND >(domain,
        grid,
        gridtools::make_multistage(execute< forward >(), gridtools::make_stage< copy_functor >(p_in(), p_out())));
    stencil_->ready();
}

void mini_dycore::copy_stencil() {
    LOG_BEGIN("mini_dycore::copy_stencil()");

    LOG(info) << "stencil_->reassign()";
    stencil_->reassign(p_in() = repository_->in(), p_out() = repository_->out());
    LOG(info) << "stencil_->steady()";
    stencil_->steady();
    LOG(info) << "stencil_->run()";
    stencil_->run();
    LOG_END();
}

GT_EXPORT_BINDING_WITH_SIGNATURE_1(copy_stencil, void(mini_dycore &), std::mem_fn(&mini_dycore::copy_stencil));

void mini_dycore::put_a_number(int number) {
    just_a_collection_of_numbers.push_back(number);
    std::cout << just_a_collection_of_numbers.size() << std::endl;
}
GT_EXPORT_BINDING_WITH_SIGNATURE_2(put_a_number, void(mini_dycore &, int), std::mem_fn(&mini_dycore::put_a_number));

void mini_dycore::print_numbers() {
    std::cout << "numbers: ";
    for (auto n : just_a_collection_of_numbers) {
        std::cout << "\n" << n;
    }
    std::cout << std::endl;
}
GT_EXPORT_BINDING_WITH_SIGNATURE_1(print_numbers, void(mini_dycore &), std::mem_fn(&mini_dycore::print_numbers));

mini_dycore alloc_mini_dycore_impl(int ndims, int *dims, std::shared_ptr< wrapped_dycore_repository > repo) {
    std::vector< gridtools::uint_t > sizes(ndims);
    for (size_t i = 0; i < ndims; ++i) {
        sizes[i] = dims[i];
    }
    return mini_dycore(sizes, repo);
}

GT_EXPORT_BINDING_3(alloc_mini_dycore, alloc_mini_dycore_impl);

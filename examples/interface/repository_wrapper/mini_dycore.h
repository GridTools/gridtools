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
#include <utility>
#include <interface/wrapper/repository_wrapper.hpp>

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

using namespace gridtools;
using namespace enumtype;

using storage_info_t = gridtools::storage_traits< BACKEND_ARCH >::storage_info_t< 0, 3 >;
using data_store_t = gridtools::storage_traits< BACKEND_ARCH >::data_store_t< double, storage_info_t >;

struct copy_functor {

    typedef accessor< 0, enumtype::in, extent<>, 3 > in;
    typedef accessor< 1, enumtype::inout, extent<>, 3 > out;
    typedef boost::mpl::vector< in, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        eval(out()) = eval(in());
    }
};

#define MY_FIELDTYPES (data_store_t, (0, 1, 2))
#define MY_FIELDS (data_store_t, in)(data_store_t, out)
GRIDTOOLS_MAKE_REPOSITORY(bare_dycore_repository, MY_FIELDTYPES, MY_FIELDS)
#undef MY_FIELDTYPES
#undef MY_FIELDS

class wrapped_dycore_repository : public repository_wrapper< bare_dycore_repository > {
  public:
    using super = repository_wrapper< bare_dycore_repository >;

    virtual ~wrapped_dycore_repository() = default;
    wrapped_dycore_repository(std::vector< uint_t > sizes) : super(sizes[0], sizes[1], sizes[2]) { Logging::enable(); }

    void run() override; // TODO not used in this example, will probably be removed in a next release
};

// TODO in a next release the the "run" will be removed and the repository will not be abstract anymore
// then the following line should be the better way and the boiler plate above will be avoided:
// using wrapped_dycore_repository = repository_wrapper< bare_dycore_repository >;

/**
 * A dycore which predicts that the weather will be the same tomorrow as it is now.
 */
class mini_dycore {
  public:
    mini_dycore(std::vector< uint_t > sizes, std::shared_ptr< wrapped_dycore_repository >);
    void copy_stencil();
    void put_a_number(int);
    void print_numbers();

  private:
    std::shared_ptr< wrapped_dycore_repository > repository_;
    using p_out = arg< 0, data_store_t >;
    using p_in = arg< 1, data_store_t >;
    using aggregator_t = gridtools::aggregator_type< boost::mpl::vector< p_out, p_in > >;
    std::shared_ptr< gridtools::computation< aggregator_t, gridtools::notype > > stencil_;

    std::vector< int > just_a_collection_of_numbers; // to illustrate what the user can do
};

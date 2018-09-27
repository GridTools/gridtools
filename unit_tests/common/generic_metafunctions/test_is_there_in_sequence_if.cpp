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
#include <gridtools/common/generic_metafunctions/is_there_in_sequence_if.hpp>

#include <type_traits>

#include <boost/mpl/vector.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>

using namespace gridtools;

TEST(is_there_in_sequence_if, is_there_in_sequence_if) {
    typedef boost::mpl::vector<int, float, char> seq_t;

    GRIDTOOLS_STATIC_ASSERT((is_there_in_sequence_if<seq_t, std::is_same<boost::mpl::_, char>>::value), "ERROR");
    GRIDTOOLS_STATIC_ASSERT((is_there_in_sequence_if<seq_t, std::is_same<boost::mpl::_, int>>::value), "ERROR");
    GRIDTOOLS_STATIC_ASSERT((!is_there_in_sequence_if<seq_t, std::is_same<boost::mpl::_, long>>::value), "ERROR");
}

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

#include <boost/mpl/minus.hpp>
#include <boost/mpl/plus.hpp>
#include "loopintervals.hpp"
#include "../common/halo_descriptor.hpp"
#include "../common/gpu_clone.hpp"
#include "storage/partitioner.hpp"
/**@file
@brief file containing the size of the horizontal domain

The domain is currently described in terms of 2 horiozntal axis of type \ref gridtools::halo_descriptor , and the
vertical axis bounds which are treated separately.
TODO This should be easily generalizable to arbitrary dimensions
*/
namespace gridtools {
    //    template < typename MinLevel, typename MaxLevel >
    //    struct make_axis {
    //        typedef interval< MinLevel, MaxLevel > type;
    //    };
    //
    //    template < typename Axis, uint_t I >
    //    struct extend_by {
    //        typedef interval< level< Axis::FromLevel::Splitter::value, Axis::FromLevel::Offset::value - 1 >,
    //            level< Axis::ToLevel::Splitter::value, Axis::ToLevel::Offset::value + 1 > > type;
    //    };

    namespace enumtype_axis {
        enum coordinate_argument { minus, plus, begin, end, length };
    } // namespace enumtype_axis

    using namespace enumtype_axis;

} // namespace gridtools

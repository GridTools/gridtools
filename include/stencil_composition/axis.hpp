/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
    template < typename MinLevel, typename MaxLevel >
    struct make_axis {
        typedef interval< MinLevel, MaxLevel > type;
    };

    template < typename Axis, uint_t I >
    struct extend_by {
        typedef interval< level< Axis::FromLevel::Splitter::value, Axis::FromLevel::Offset::value - 1 >,
            level< Axis::ToLevel::Splitter::value, Axis::ToLevel::Offset::value + 1 > > type;
    };

    namespace enumtype_axis {
        enum coordinate_argument { minus, plus, begin, end, length };
    } // namespace enumtype_axis

    using namespace enumtype_axis;

} // namespace gridtools

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
#include "mss.hpp"
#include "reductions/reduction_descriptor.hpp"

namespace gridtools {
    /**
     * type traits for a-mss descriptor. Amss descriptor is any descriptor that implements the concept
     * a MSS: currently mss_descriptor and reduction_descriptor
     */
    template < typename T >
    struct amss_descriptor_is_reduction;

    template < typename ExecutionEngine, typename EsfDescrSequence >
    struct amss_descriptor_is_reduction< mss_descriptor< ExecutionEngine, EsfDescrSequence > > : boost::mpl::false_ {};

    template < typename ReductionType, typename BinOp, typename EsfDescrSequence >
    struct amss_descriptor_is_reduction< reduction_descriptor< ReductionType, BinOp, EsfDescrSequence > >
        : boost::mpl::true_ {};
}

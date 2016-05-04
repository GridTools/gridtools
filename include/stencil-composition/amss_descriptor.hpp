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

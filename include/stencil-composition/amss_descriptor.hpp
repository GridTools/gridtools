#pragma once

namespace gridtools {
    template < typename T >
    struct is_amss_descriptor : boost::mpl::false_ {};

    template < typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence >
    struct is_amss_descriptor< mss_descriptor< ExecutionEngine, EsfDescrSequence, CacheSequence > >
        : boost::mpl::true_ {};

    template < typename ReductionType, enumtype::binop BinOp, typename EsfDescrSequence >
    struct is_amss_descriptor< reduction_descriptor< ReductionType, BinOp, EsfDescrSequence > >
        : boost::mpl::true_ {};

    template < typename T >
    struct amss_descriptor_is_reduction;

    template < typename ExecutionEngine, typename EsfDescrSequence >
    struct amss_descriptor_is_reduction< mss_descriptor< ExecutionEngine, EsfDescrSequence > >
        : boost::mpl::false_ {};

    template < typename ReductionType, enumtype::binop BinOp, typename EsfDescrSequence >
    struct amss_descriptor_is_reduction< reduction_descriptor< ReductionType, BinOp, EsfDescrSequence> >
        : boost::mpl::true_ {};
}

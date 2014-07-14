#pragma once
#include <boost/mpl/assert.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/push_back.hpp>

#include "stencil-composition/esf.h"
#include "stencil-composition/mss.h"

namespace gridtools {

    template <typename ESF,
              typename A0>
    esf_descriptor<ESF, boost::mpl::vector1<A0> >
    make_esf(A0) {
        BOOST_MPL_ASSERT_RELATION(boost::mpl::size<typename ESF::arg_list>::value, ==, 1);
        return esf_descriptor<ESF, boost::mpl::vector1<A0> >();
    }

    template <typename ESF,
              typename A0,
              typename A1>
    esf_descriptor<ESF, boost::mpl::vector2<A0, A1> >
    make_esf(A0, A1) {
        BOOST_MPL_ASSERT_RELATION(boost::mpl::size<typename ESF::arg_list>::value, ==, 2);
        return esf_descriptor<ESF, boost::mpl::vector2<A0, A1> >();
    }

    template <typename ESF,
              typename A0,
              typename A1,
              typename A2>
    esf_descriptor<ESF, boost::mpl::vector3<A0, A1, A2> >
    make_esf(A0, A1, A2) {
        BOOST_MPL_ASSERT_RELATION(boost::mpl::size<typename ESF::arg_list>::value, ==, 3);
        return esf_descriptor<ESF, boost::mpl::vector3<A0, A1, A2> >();
    }

    template <typename ESF,
              typename A0,
              typename A1,
              typename A2,
              typename A3>
    esf_descriptor<ESF, boost::mpl::vector4<A0, A1, A2, A3> >
    make_esf(A0, A1, A2, A3) {
        BOOST_MPL_ASSERT_RELATION(boost::mpl::size<typename ESF::arg_list>::value, ==, 4);
        return esf_descriptor<ESF, boost::mpl::vector4<A0, A1, A2, A3> >();
    }

    template <typename ESF,
              typename A0,
              typename A1,
              typename A2,
              typename A3,
              typename A4>
    esf_descriptor<ESF, boost::mpl::vector5<A0, A1, A2, A3, A4> >
    make_esf(A0, A1, A2, A3, A4) {
        BOOST_MPL_ASSERT_RELATION(boost::mpl::size<typename ESF::arg_list>::value, ==, 5);
        return esf_descriptor<ESF, boost::mpl::vector5<A0, A1, A2, A3, A4> >();
    }

    template <typename ExecutionEngine,
              typename EsfDescr0>
    mss_descriptor<ExecutionEngine, boost::mpl::vector1<EsfDescr0> >
    make_mss(ExecutionEngine const&, EsfDescr0 const&) {
        return mss_descriptor<ExecutionEngine, boost::mpl::vector1<EsfDescr0> >();
    }

    template <typename ExecutionEngine,
              typename EsfDescr0,
              typename EsfDescr1>
    mss_descriptor<ExecutionEngine, boost::mpl::vector2<EsfDescr0, EsfDescr1> >
    make_mss(ExecutionEngine const&, EsfDescr0 const&, EsfDescr1 const&) {
        return mss_descriptor<ExecutionEngine, boost::mpl::vector2<EsfDescr0, EsfDescr1> >();
    }

    template <typename ExecutionEngine,
              typename EsfDescr0,
              typename EsfDescr1,
              typename EsfDescr2>
    mss_descriptor<ExecutionEngine, boost::mpl::vector3<EsfDescr0, EsfDescr1, EsfDescr2> >
    make_mss(ExecutionEngine const&, EsfDescr0 const&, EsfDescr1 const&, EsfDescr2 const&) {
        return mss_descriptor<ExecutionEngine, boost::mpl::vector3<EsfDescr0, EsfDescr1, EsfDescr2> >();
    }

    template <typename ExecutionEngine,
              typename EsfDescr0,
              typename EsfDescr1,
              typename EsfDescr2,
              typename EsfDescr3>
    mss_descriptor<ExecutionEngine, boost::mpl::vector4<EsfDescr0, EsfDescr1, EsfDescr2, EsfDescr3> >
    make_mss(ExecutionEngine const&, EsfDescr0 const&, EsfDescr1 const&, EsfDescr2 const&, EsfDescr3 const&) {
        return mss_descriptor<ExecutionEngine, boost::mpl::vector4<EsfDescr0, EsfDescr1, EsfDescr2, EsfDescr3> >();
    }

    template <typename ExecutionEngine,
              typename EsfDescr0,
              typename EsfDescr1,
              typename EsfDescr2,
              typename EsfDescr3,
              typename EsfDescr4>
    mss_descriptor<ExecutionEngine, boost::mpl::vector5<EsfDescr0, EsfDescr1, EsfDescr2, EsfDescr3, EsfDescr4> >
    make_mss(ExecutionEngine const&, EsfDescr0 const&, EsfDescr1 const&, EsfDescr2 const&, EsfDescr3 const&, EsfDescr4 const&) {
        return mss_descriptor<ExecutionEngine, boost::mpl::vector5<EsfDescr0, EsfDescr1, EsfDescr2, EsfDescr3, EsfDescr4> >();
    }


    template <typename EsfDescr0,
              typename EsfDescr1>
    independent_esf<boost::mpl::vector2<EsfDescr0, EsfDescr1> >
    make_independent(EsfDescr0 const&, EsfDescr1 const&) {
        return independent_esf<boost::mpl::vector2<EsfDescr0, EsfDescr1> >();
    }
} // namespace gridtools

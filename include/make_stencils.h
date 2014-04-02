#pragma once
#include <boost/mpl/assert.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/push_back.hpp>

namespace gridtools {
    template <int N_ARGS,
              typename A0 = void,
              typename A1 = void,
              typename A2 = void,
              typename A3 = void,
              typename A4 = void>
    struct with_wrapper;

    template <typename A0>
    struct with_wrapper<1, A0> {
        typedef boost::mpl::vector1<A0> result_type;

        static
        result_type
        with(A0 &) {
            return result_type();
        }
    };

    template <typename A0, typename A1>
    struct with_wrapper<2, A0, A1> {
        typedef boost::mpl::vector2<A0, A1> result_type;

        static
        result_type
        with(A0 &, A1&) {
            return result_type();
        }
    };

    template <typename A0, typename A1, typename A2>
    struct with_wrapper<3, A0, A1, A2> {
        typedef boost::mpl::vector3<A0, A1, A2> result_type;

        static
        result_type
        with(A0 &, A1&, A2&) {
            return result_type();
        }
    };

    template <typename A0, typename A1, typename A2, typename A3>
    struct with_wrapper<4, A0, A1, A2, A3> {
        typedef boost::mpl::vector4<A0, A1, A2, A3> result_type;

        static
        result_type
        with(A0 &, A1&, A2&, A3&) {
            return result_type();
        }
    };

    template <typename A0, typename A1, typename A2, typename A3, typename A4>
    struct with_wrapper<5, A0, A1, A2, A3, A4> {
        typedef boost::mpl::vector5<A0, A1, A2, A3, A4> result_type;

        static
        result_type
        with(A0 &, A1&, A2&, A3&) {
            return result_type();
        }
    };



    // Descriptors for ESF
    template <typename ESF, typename ArgArray>
    struct esf_descriptor {
        typedef ESF esf_function;
        typedef ArgArray args;
    };

    template <typename T, typename V>
    std::ostream& operator<<(std::ostream& s, esf_descriptor<T,V> const7) {
        return s << "esf_desctiptor< " << T() << " , somevector > ";
    }

    // Descriptors for ES
    template <typename ExecutionEngine,
              typename EsfDescr>
    struct es_descriptor {};

    template <typename ArgArray>
    struct independent_esf {
        typedef ArgArray esf_list;
    };

    template <typename T>
    struct is_independent 
      : boost::false_type
    {};

    template <typename T>
    struct is_independent<independent_esf<T> >
      : boost::true_type
    {};


    // Descriptors for MSS
    template <typename ExecutionEngine,
              typename ArrayEsfDescr>
    struct mss_descriptor {
        template <typename State, typename SubArray>
        struct keep_scanning
          : boost::mpl::fold<
                typename SubArray::esf_list,
                State,
                boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>
            >
        {};

        template <typename Array>
        struct linearize_esf_array
          : boost::mpl::fold<
                Array,
                boost::mpl::vector<>,
                boost::mpl::if_<
                    is_independent<boost::mpl::_2>,
                    keep_scanning<boost::mpl::_1, boost::mpl::_2>,
                    boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>
                >
            >
        {};

        typedef ArrayEsfDescr esf_array; // may contain independent constructs
        typedef ExecutionEngine execution_engine;

        // Collect all esf nodes in the the multi-stage descriptor. Recurse into independent
        // esf structs. Independent functors are listed one after the other.
        typedef typename linearize_esf_array<esf_array>::type linear_esf;
    };


    template <typename ESF,
              typename A0>
    esf_descriptor<ESF, typename with_wrapper<1, A0>::result_type >
    make_esf(A0) {
        BOOST_MPL_ASSERT_RELATION(boost::mpl::size<typename ESF::arg_list>::value, ==, 1);
        return esf_descriptor<ESF, typename with_wrapper<1, A0>::result_type >();
    }

    template <typename ESF,
              typename A0,
              typename A1>
    esf_descriptor<ESF, typename with_wrapper<2, A0, A1>::result_type >
    make_esf(A0, A1) {
        BOOST_MPL_ASSERT_RELATION(boost::mpl::size<typename ESF::arg_list>::value, ==, 2);
        return esf_descriptor<ESF, typename with_wrapper<2, A0, A1>::result_type >();
    }

    template <typename ESF,
              typename A0,
              typename A1,
              typename A2>
    esf_descriptor<ESF, typename with_wrapper<3, A0, A1, A2>::result_type >
    make_esf(A0, A1, A2) {
        BOOST_MPL_ASSERT_RELATION(boost::mpl::size<typename ESF::arg_list>::value, ==, 3);
        return esf_descriptor<ESF, typename with_wrapper<3, A0, A1, A2>::result_type >();
    }

    template <typename ESF,
              typename A0,
              typename A1,
              typename A2,
              typename A3>
    esf_descriptor<ESF, typename with_wrapper<4, A0, A1, A2, A3>::result_type >
    make_esf(A0, A1, A2, A3) {
        BOOST_MPL_ASSERT_RELATION(boost::mpl::size<typename ESF::arg_list>::value, ==, 4);
        return esf_descriptor<ESF, typename with_wrapper<4, A0, A1, A2, A3>::result_type >();
    }

    template <typename ESF,
              typename A0,
              typename A1,
              typename A2,
              typename A3,
              typename A4>
    esf_descriptor<ESF, typename with_wrapper<5, A0, A1, A2, A3, A4>::result_type >
    make_esf(A0, A1, A2, A3, A4) {
        BOOST_MPL_ASSERT_RELATION(boost::mpl::size<typename ESF::arg_list>::value, ==, 5);
        return esf_descriptor<ESF, typename with_wrapper<5, A0, A1, A2, A3, A4>::result_type >();
    }

    template <typename ExecutionEngine,
              typename EsfDescr>
    es_descriptor<ExecutionEngine, EsfDescr>
    make_es(ExecutionEngine const&, EsfDescr const&) {
        return es_descriptor<ExecutionEngine, EsfDescr>();
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

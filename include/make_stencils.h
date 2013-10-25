#pragma once
#include <boost/mpl/vector.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/push_back.hpp>

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
template <typename ESF, typename t_arg_array>
struct esf_descriptor {
    typedef ESF esf_function;
    typedef t_arg_array args;
};

// Descriptors for ES
template <typename t_execution_engine,
          typename t_esf_descr>
struct es_descriptor {};

template <typename t_arg_array>
struct independent_esf {
    typedef  t_arg_array esf_list;
};

template <typename T>
struct is_independent {
    typedef boost::false_type type;
    static const bool value = false;
};

template <typename T>
struct is_independent<independent_esf<T> > {
    typedef boost::true_type type;
    static const bool value = true;
};


// Descriptors for MSS
template <typename t_execution_engine,
          typename t_array_esf_descr>
struct mss_descriptor {
    template <typename state, typename subarray>
    struct keep_scanning {
        typedef typename boost::mpl::fold<
            typename subarray::esf_list,
            state,
            boost::mpl::push_back<boost::mpl::_1,boost::mpl::_2>
            >::type type;
    };

    template <typename array>
    struct linearize_esf_array {
        typedef typename boost::mpl::fold<array,
                                          boost::mpl::vector<>,
                                          boost::mpl::if_<
                                              is_independent<boost::mpl::_2>,
                                              keep_scanning<boost::mpl::_1, boost::mpl::_2>,
                                              boost::mpl::push_back<boost::mpl::_1,boost::mpl::_2>
                                              >
                                          >::type type;
    };

    typedef t_array_esf_descr esf_array; // may contain independent constructs
    typedef t_execution_engine execution_engine;

    typedef typename linearize_esf_array<esf_array>::type linear_esf; // independent functors are listed one after the other

};


template <typename ESF,
          typename A0>
esf_descriptor<ESF, typename with_wrapper<1, A0>::result_type >
make_esf(A0) {
    return esf_descriptor<ESF, typename with_wrapper<1, A0>::result_type >();
}

template <typename ESF,
          typename A0,
          typename A1>
esf_descriptor<ESF, typename with_wrapper<2, A0, A1>::result_type >
make_esf(A0, A1) {
    return esf_descriptor<ESF, typename with_wrapper<2, A0, A1>::result_type >();
}

template <typename ESF,
          typename A0,
          typename A1,
          typename A2>
esf_descriptor<ESF, typename with_wrapper<3, A0, A1, A2>::result_type >
make_esf(A0, A1, A2) {
    return esf_descriptor<ESF, typename with_wrapper<3, A0, A1, A2>::result_type >();
}

template <typename ESF,
          typename A0,
          typename A1,
          typename A2,
          typename A3>
esf_descriptor<ESF, typename with_wrapper<4, A0, A1, A2, A3>::result_type >
make_esf(A0, A1, A2, A3) {
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
    return esf_descriptor<ESF, typename with_wrapper<5, A0, A1, A2, A3, A4>::result_type >();
}

template <typename t_execution_engine,
          typename t_esf_descr>
es_descriptor<t_execution_engine, t_esf_descr>
make_es(t_execution_engine const&, t_esf_descr const&) {
    return es_descriptor<t_execution_engine, t_esf_descr>();
}

template <typename t_execution_engine,
          typename t_esf_descr0>
mss_descriptor<t_execution_engine, boost::mpl::vector1<t_esf_descr0> >
make_mss(t_execution_engine const&, t_esf_descr0 const&) {
    return mss_descriptor<t_execution_engine, boost::mpl::vector1<t_esf_descr0> >();
}

template <typename t_execution_engine,
          typename t_esf_descr0,
          typename t_esf_descr1>
mss_descriptor<t_execution_engine, boost::mpl::vector2<t_esf_descr0, t_esf_descr1> >
make_mss(t_execution_engine const&, t_esf_descr0 const&, t_esf_descr1 const&) {
    return mss_descriptor<t_execution_engine, boost::mpl::vector2<t_esf_descr0, t_esf_descr1> >();
}

template <typename t_execution_engine,
          typename t_esf_descr0,
          typename t_esf_descr1,
          typename t_esf_descr2>
mss_descriptor<t_execution_engine, boost::mpl::vector3<t_esf_descr0, t_esf_descr1, t_esf_descr2> >
make_mss(t_execution_engine const&, t_esf_descr0 const&, t_esf_descr1 const&, t_esf_descr2 const&) {
    return mss_descriptor<t_execution_engine, boost::mpl::vector3<t_esf_descr0, t_esf_descr1, t_esf_descr2> >();
}

template <typename t_execution_engine,
          typename t_esf_descr0,
          typename t_esf_descr1,
          typename t_esf_descr2,
          typename t_esf_descr3>
mss_descriptor<t_execution_engine, boost::mpl::vector4<t_esf_descr0, t_esf_descr1, t_esf_descr2, t_esf_descr3> >
make_mss(t_execution_engine const&, t_esf_descr0 const&, t_esf_descr1 const&, t_esf_descr2 const&, t_esf_descr3 const&) {
    return mss_descriptor<t_execution_engine, boost::mpl::vector4<t_esf_descr0, t_esf_descr1, t_esf_descr2, t_esf_descr3> >();
}

template <typename t_execution_engine,
          typename t_esf_descr0,
          typename t_esf_descr1,
          typename t_esf_descr2,
          typename t_esf_descr3,
          typename t_esf_descr4>
mss_descriptor<t_execution_engine, boost::mpl::vector5<t_esf_descr0, t_esf_descr1, t_esf_descr2, t_esf_descr3, t_esf_descr4> >
make_mss(t_execution_engine const&, t_esf_descr0 const&, t_esf_descr1 const&, t_esf_descr2 const&, t_esf_descr3 const&, t_esf_descr4 const&) {
    return mss_descriptor<t_execution_engine, boost::mpl::vector5<t_esf_descr0, t_esf_descr1, t_esf_descr2, t_esf_descr3, t_esf_descr4> >();
}


template <typename t_esf_descr0,
          typename t_esf_descr1>
independent_esf<boost::mpl::vector2<t_esf_descr0, t_esf_descr1> >
make_independent(t_esf_descr0 const&, t_esf_descr1 const&) {
    return independent_esf<boost::mpl::vector2<t_esf_descr0, t_esf_descr1> >();
}

#pragma once

#include "storage.h"
#include "layout_map.h"
#include "range.h"
#include <boost/type_traits/integral_constant.hpp>

template <typename t_value_type>
struct temporary {
    typedef t_value_type value_type;
    //    typedef t_value_type value_type;
};


template <typename T>
struct is_temporary {
    typedef boost::false_type type;
};

template <typename U>
struct is_temporary<temporary<U> > {
    typedef boost::true_type type;
};    

template <int I, typename T>
struct arg {
    //        int m_i,m_j,m_k;
    typedef T storage_type;
    typedef typename T::iterator_type iterator_type;
    typedef typename T::value_type value_type;
    typedef boost::mpl::int_<I> index_type;
    typedef boost::mpl::int_<I> index;
    //    static const int index = I;

    // arg(int i, int j, int k)
    //     : m_i(i)
    //     , m_j(j)
    //     , m_k(k)
    // {}

    // arg()
    //     : m_i(0)
    //     , m_j(0)
    //     , m_k(0)
    // {}

    static arg<I,T> center() {
        return arg<I,T>();
    }
};

template <int I, typename U>
struct arg<I, temporary<U> > {
    typedef U value_type;
    typedef storage<U, GCL::layout_map<0,1,2>, true, arg<I, temporary<U> > > storage_type;
    typedef typename storage_type::iterator_type iterator_type;
    typedef boost::mpl::int_<I> index_type;
    typedef boost::mpl::int_<I> index;
//    static const int index = I;
};

template <int I, typename t_range=range<0,0,0,0> >
struct arg_type {

    template <int im, int ip, int jm, int jp, int kp, int km>
    struct halo {
        typedef arg_type<I> type;
    };

    int offset[3];
    typedef typename boost::mpl::int_<I> index_type;
    typedef typename boost::mpl::int_<I> index;
    typedef t_range range_type;

    arg_type(int i, int j, int k) {
        offset[0] = i;
        offset[1] = j;
        offset[2] = k;
    }

    arg_type() {
        offset[0] = 0;
        offset[1] = 0;
        offset[2] = 0;
    }

    int i() const {
        return offset[0];
    }

    int j() const {
        return offset[1];
    }
    
    int k() const {
        return offset[2];
    }

    static arg_type<I> center() {
        return arg_type<I>();
    }

    const int* offset_ptr() const {
        return offset;
    }

    arg_type<I> plus(int _i, int _j, int _k) const {
        return arg_type<I>(i()+_i, j()+_j, k()+_k);
    }
};

template <int I, typename R>
std::ostream& operator<<(std::ostream& s, arg_type<I,R> const&) {
    return s << "[ arg_type< " << I
             << ", " << R() << " > ]";
}

template <int I, typename R>
std::ostream& operator<<(std::ostream& s, arg<I,temporary<R> > const&) {
    return s << "[ arg< " << I
             << ", temporary<something>" << " > ]";
}

template <int I, typename R>
std::ostream& operator<<(std::ostream& s, arg<I,R> const&) {
    return s << "[ arg< " << I
             << ", NON TEMP" << " > ]";
}

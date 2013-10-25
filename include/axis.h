#pragma once 

#include <boost/mpl/minus.hpp>
#include <boost/mpl/plus.hpp>
#include "LoopIntervals.h"
#include "array.h"

template <typename t_min_level, typename t_max_level>
struct make_axis {
    typedef Interval<t_min_level, t_max_level> type;
};

template <typename t_axis, int I>
struct extend_by {
    typedef Interval<Level<t_axis::FromLevel::Splitter::value, t_axis::FromLevel::Offset::value - 1>,
                     Level<t_axis::ToLevel::Splitter::value, t_axis::ToLevel::Offset::value + 1> > type;
};

template <typename t_axis>
struct coordinates {
    BOOST_STATIC_ASSERT(is_interval<t_axis>::value);

    typedef t_axis axis_type;

    GCL::array<int, boost::mpl::plus<
                        typename boost::mpl::minus<typename t_axis::ToLevel::Splitter, 
                                                   typename t_axis::FromLevel::Splitter>,
                        typename boost::mpl::int_<1> >::value > value_list;
    int i_low_bound;
    int i_high_bound;
    int j_low_bound;
    int j_high_bound;

    explicit coordinates(int il, int ih, int jl, int jh)
        : i_low_bound(il)
        , i_high_bound(ih)
        , j_low_bound(jl)
        , j_high_bound(jh)
    {}
    
    template <typename t_level>
    int value_at() const {
        BOOST_STATIC_ASSERT(is_level<t_level>::value);
        int offs = t_level::Offset::value;
        if (offs < 0) offs += 1;
        return value_list[t_level::Splitter::value] + offs;
    }
};


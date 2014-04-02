#pragma once 

#include <boost/mpl/minus.hpp>
#include <boost/mpl/plus.hpp>
#include "loopintervals.h"
#include "array.h"
#include "gpu_clone.h"

namespace gridtools {
    template <typename MinLevel, typename MaxLevel>
    struct make_axis {
        typedef interval<MinLevel, MaxLevel> type;
    };

    template <typename Axis, int I>
    struct extend_by {
        typedef interval<level<Axis::FromLevel::Splitter::value, Axis::FromLevel::Offset::value - 1>,
                         level<Axis::ToLevel::Splitter::value, Axis::ToLevel::Offset::value + 1> > type;
    };

    template <typename Axis>
    struct coordinates : public clonable_to_gpu<coordinates<Axis> > {
        BOOST_STATIC_ASSERT(is_interval<Axis>::value);

        typedef Axis axis_type;

    typedef typename boost::mpl::plus<
        boost::mpl::minus<typename Axis::ToLevel::Splitter, 
                          typename Axis::FromLevel::Splitter>,
        boost::mpl::int_<1> >::type size_type;
    
        gridtools::array<int, size_type::value > value_list;
    
        int _i_low_bound;
        int _i_high_bound;
        int _j_low_bound;
        int _j_high_bound;

        GT_FUNCTION
        explicit coordinates(int il, int ih, int jl, int jh)
            : _i_low_bound(il)
            , _i_high_bound(ih)
            , _j_low_bound(jl)
            , _j_high_bound(jh)
        {}
        
        GT_FUNCTION
        int i_low_bound() const {
            return _i_low_bound;
        }

        GT_FUNCTION
        int i_high_bound() const {
            return _i_high_bound;
        }

        GT_FUNCTION
        int j_low_bound() const {
            return _j_low_bound;
        }

        GT_FUNCTION
        int j_high_bound() const {
            return _j_high_bound;
        }

        template <typename Level>
        GT_FUNCTION
        int value_at() const {
            BOOST_STATIC_ASSERT(is_level<Level>::value);
            int offs = Level::Offset::value;
            if (offs < 0) offs += 1;
            return value_list[Level::Splitter::value] + offs;
        }

        template <typename Level>
        GT_FUNCTION
        int& value_at(int val) const {
            BOOST_STATIC_ASSERT(is_level<Level>::value);
            return value_list[Level::Splitter::value];
        }

        GT_FUNCTION
        int value_at_top() const {
            return value_list[size_type::value - 1];
        }

        GT_FUNCTION
        int value_at_bottom() const {
            return value_list[0];
        }
    };
} // namespace gridtools

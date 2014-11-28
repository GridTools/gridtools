#pragma once

#include <boost/mpl/minus.hpp>
#include <boost/mpl/plus.hpp>
#include "loopintervals.h"
#include "../common/halo_descriptor.h"
#include "../common/gpu_clone.h"

#include <boost/mpl/print.hpp>

namespace gridtools {
    template <typename MinLevel, typename MaxLevel>
    struct make_axis {
        typedef interval<MinLevel, MaxLevel> type;
    };

    template <typename Axis, uint_t I>
    struct extend_by {
        typedef interval<level<Axis::FromLevel::Splitter::value, Axis::FromLevel::Offset::value - 1>,
                         level<Axis::ToLevel::Splitter::value, Axis::ToLevel::Offset::value + 1> > type;
    };

    namespace    enumtype_axis{
        enum coordinate_argument {minus, plus, begin, end, length};
    }//namespace enumtype_axis

    using namespace enumtype_axis;

    template <typename Axis>
    struct coordinates : public clonable_to_gpu<coordinates<Axis> > {
        BOOST_STATIC_ASSERT(is_interval<Axis>::value);


        typedef Axis axis_type;

    typedef typename boost::mpl::plus<
        boost::mpl::minus<typename Axis::ToLevel::Splitter,
                          typename Axis::FromLevel::Splitter>,
        static_int<1> >::type size_type;

        gridtools::array<uint_t, size_type::value > value_list;

        GT_FUNCTION
        explicit coordinates(uint_t il, uint_t ih, uint_t jl, uint_t jh/*, uint_t kl, uint_t kh*/)
            : m_i_low_bound(il)
            , m_i_high_bound(ih)
            , m_j_low_bound(jl)
            , m_j_high_bound(jh)
            , m_direction_i()
            , m_direction_j()
        {}

        GT_FUNCTION
        explicit coordinates( uint_t* i, uint_t* j/*, uint_t* k*/)
            : m_i_low_bound(i[begin])
            , m_i_high_bound(i[end])
            , m_j_low_bound(j[begin])
            , m_j_high_bound(j[end])
            // , m_k_low_bound(k[begin])
            // , m_k_high_bound(k[end])
            , m_direction_i(i[minus], i[plus], i[begin], i[end], i[length])
            , m_direction_j(j[minus], j[plus], j[begin], j[end], j[length])
            // , m_direction_k(k[minus], k[plus], k[begin], k[end], k[length])
        {}

        GT_FUNCTION
        uint_t i_low_bound() const {
            return m_i_low_bound;
        }

        GT_FUNCTION
        uint_t i_high_bound() const {
            return m_i_high_bound;
        }

        GT_FUNCTION
        uint_t j_low_bound() const {
            return m_j_low_bound;
        }

        GT_FUNCTION
        uint_t j_high_bound() const {
            return m_j_high_bound;
        }

        template <typename Level>
        GT_FUNCTION
        uint_t value_at() const {
            BOOST_STATIC_ASSERT(is_level<Level>::value);
            int_t offs = Level::Offset::value;
            if (offs < 0) offs += 1;
            return value_list[Level::Splitter::value] + offs;
        }

        // template <typename Level>
        // GT_FUNCTION
        // uint_t& value_at(uint_t val) const {
        //     BOOST_STATIC_ASSERT(is_level<Level>::value);
        //     return value_list[Level::Splitter::value];
        // }

        GT_FUNCTION
        uint_t value_at_top() const {
            return value_list[size_type::value - 1];
            // return m_k_high_bound;
        }

        GT_FUNCTION
        uint_t value_at_bottom() const {
            return value_list[0];
            // return m_k_low_bound;
        }

        gridtools::halo_descriptor const& direction_i() const { return m_direction_i;}

        gridtools::halo_descriptor const& direction_j() const { return m_direction_j;}

    private:

        gridtools::halo_descriptor m_direction_i;
        gridtools::halo_descriptor m_direction_j;
        // gridtools::halo_descriptor m_direction_k;

        uint_t m_i_low_bound;
        uint_t m_i_high_bound;
        uint_t m_j_low_bound;
        uint_t m_j_high_bound;
        // int m_k_low_bound;
        // int m_k_high_bound;

    };
} // namespace gridtools

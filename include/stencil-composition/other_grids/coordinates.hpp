#pragma once

namespace gridtools {

template <typename Axis, typename Grid>
struct coordinates : public clonable_to_gpu<coordinates<Axis, Grid> > {
    GRIDTOOLS_STATIC_ASSERT((is_interval<Axis>::value), "Internal Error: wrong type");
    GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), "Internal Error: wrong type");

    typedef Axis axis_type;
    typedef Grid grid_t;

    typedef typename boost::mpl::plus<
            boost::mpl::minus<typename Axis::ToLevel::Splitter, typename Axis::FromLevel::Splitter>,
            static_int<1>
    >::type size_type;

    array<uint_t, size_type::value > value_list;
private:
    Grid m_grid;
    const array<uint_t,2> m_i_bounds, m_j_bounds;
public:
    GT_FUNCTION
    explicit coordinates(Grid& grid, array<uint_t, 2>& i, array<uint_t, 2>& j) :
        m_grid(grid), m_i_bounds(i), m_j_bounds(j)
    {}

    GT_FUNCTION
    uint_t i_low_bound() const {
        return m_i_bounds[0];
    }

    GT_FUNCTION
    uint_t i_high_bound() const {
        return m_i_bounds[1];
    }

    GT_FUNCTION
    uint_t j_low_bound() const {
        return m_j_bounds[0];
    }

    GT_FUNCTION
    uint_t j_high_bound() const {
        return m_j_bounds[1];
    }

    template <typename Level>
    GT_FUNCTION
    uint_t value_at() const {
        GRIDTOOLS_STATIC_ASSERT((is_level<Level>::value), "Internal Error: wrong type");
        int_t offs = Level::Offset::value;
        if (offs < 0) offs += 1;
        return value_list[Level::Splitter::value] + offs;
    }

    GT_FUNCTION
    uint_t value_at_top() const {
        return value_list[size_type::value - 1];
    }

    GT_FUNCTION
    uint_t value_at_bottom() const {
        return value_list[0];
    }

private:


};

template<typename Coord>
struct is_coordinates : boost::mpl::false_{};

template<typename Axis, typename Grid>
struct is_coordinates<coordinates<Axis, Grid> > : boost::mpl::true_{};

}

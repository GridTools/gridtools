#pragma once
#include "stencil-composition/axis.hpp"
#include "common/halo_descriptor.hpp"
#include "stencil-composition/common_grids/coordinates_cg.hpp"
#include "stencil-composition/icosahedral_grids/grid.hpp"

namespace gridtools {

    template <typename Axis, typename Grid>
    struct coordinates : public coordinates_cg<Axis>, public clonable_to_gpu<coordinates<Axis, Grid> > {
        GRIDTOOLS_STATIC_ASSERT((is_interval<Axis>::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_grid_topology<Grid>::value), "Internal Error: wrong type");

        typedef Grid grid_t;
    private:
        Grid& m_grid;
    public:
        GT_FUNCTION
        //TODO make grid const
        explicit coordinates(Grid& grid, const array<uint_t, 5>& i, const array<uint_t, 5>& j) :
            coordinates_cg<Axis>(i,j), m_grid(grid)
        {}

        Grid const & grid() const {
            return m_grid;
        }
    };

    template<typename Coord>
    struct is_coordinates : boost::mpl::false_{};

    template<typename Axis, typename Grid>
    struct is_coordinates<coordinates<Axis, Grid> > : boost::mpl::true_{};

}

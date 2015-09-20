#pragma once

#include <stencil-composition/make_computation.hpp>
// [includes]
#include "basis_functions.h"
#include "intrepid.h"
// [includes]
#ifdef CXX11_ENABLED

using namespace gridtools;
template <typename Geometry>
struct assembly_base{

    using grid_type=storage_t<gridtools::layout_map<0,1,2,3,4> >;

    typedef arg<0, grid_type >       p_grid_points;
    static const ushort_t size=1;

    using geo_map=typename Geometry::geo_map;

// [private members]
    uint_t m_d1, m_d2, m_d3;
private:

    grid_type m_grid;
// [private members]

public:

    assembly_base( uint_t d1, uint_t d2, uint_t d3 ):
        m_d1(d1)
        , m_d2(d2)
        , m_d3(d3)
        , m_grid(d1, d2, d3, geo_map::basisCardinality, 3)
        {        }

    grid_type const& get_grid() const {return m_grid;}
    grid_type& grid() {return m_grid;}

    template <typename ... MPLList>
    gridtools::domain_type< boost::mpl::vector<p_grid_points, MPLList ...> >
    domain(typename MPLList::storage_type& ...  storages_ ){
        return domain_type<boost::mpl::vector<p_grid_points, MPLList ...> >(boost::fusion::make_vector(&m_grid, &storages_ ...));
    }

}; //struct assembly


#endif //CXX11_ENABLED

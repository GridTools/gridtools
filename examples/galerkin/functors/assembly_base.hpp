#pragma once

#include <stencil-composition/make_computation.hpp>
// [includes]
#include "../numerics/basis_functions.hpp"
#include "../numerics/intrepid.hpp"
// [includes]
#ifdef CXX11_ENABLED

using namespace gridtools;
template <typename Geometry>
struct assembly_base{

    using geometry_t = Geometry;
    using grid_type_info=storage_info< layout_tt<3,4>, __COUNTER__  >;
    using grid_type=storage_t< grid_type_info >;

    using geo_map=typename Geometry::geo_map;

// [private members]
    uint_t m_d1, m_d2, m_d3;
private:

    grid_type_info m_grid_info;
    grid_type m_grid;
// [private members]

public:

    assembly_base( uint_t d1, uint_t d2, uint_t d3 ):
        m_d1(d1)
        , m_d2(d2)
        , m_d3(d3)
        , m_grid_info(d1, d2, d3, geo_map::basisCardinality, 3)
        , m_grid(m_grid_info, 0., "grid")
        {        }

    grid_type const& get_grid() const {return m_grid;}
    grid_type& grid() {return m_grid;}

}; //struct assembly


template < typename ... Types >
struct domain_type_tuple;

template < typename Geometry >
struct domain_type_tuple<assembly_base<Geometry> >{

private:
    using as_t = assembly_base<Geometry>;
    as_t & m_as;

public:

    domain_type_tuple(as_t & as_) : m_as(as_) {}

    /**I have to define here the placeholders to the storages used: the temporary storages get internally managed, while
       non-temporary ones must be instantiated by the user. In this example all the storages are non-temporaries.*/

    typedef arg<0, typename as_t::grid_type >       p_grid_points;
    static const ushort_t size=1;

    template <typename ... MPLList>
    gridtools::domain_type< boost::mpl::vector<p_grid_points, MPLList ...> >
    domain(typename MPLList::storage_type& ...  storages_ ){
        return domain_type<boost::mpl::vector<p_grid_points, MPLList ...> >(boost::fusion::make_vector(&m_as.grid(), &storages_ ...));
    }

};

#endif //CXX11_ENABLED

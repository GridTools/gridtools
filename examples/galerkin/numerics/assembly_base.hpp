#pragma once

#include <stencil-composition/stencil-composition.hpp>
// [includes]
#include "basis_functions.hpp"
#include "../functors/assembly_functors.hpp"
#include "intrepid.hpp"
// [includes]
#ifdef CXX11_ENABLED

// template <typename ... Args>
// struct make_domain_type{
//     typedef typename domain_type<boost::mpl::vector<Args ... > > type;
// };

// template<typename DomainArgs ..., typename PlaceHolders ...>
// struct concatenate<make_domain_type<DomainArgs ...>, PlaceHolders ...>{
//     typedef typename make_domain_type<DomainArgs ..., PlaceHolders ...> type;
// }



using namespace gridtools;
template <typename Geometry>
struct assembly_base{

    using geometry_t = Geometry;
    using grid_type_info=storage_info< __COUNTER__, layout_tt<3,4>  >;
    using grid_map_type_info=storage_info< __COUNTER__, layout_tt<3>  >;
    using grid_type=storage_t< grid_type_info >;
    using grid_map_type=storage_t< grid_map_type_info >;

    using geo_map=typename Geometry::geo_map;

// [private members]
    uint_t m_d1, m_d2, m_d3;
private:

    grid_type_info m_grid_info;
    grid_type m_grid;
    grid_map_type_info m_grid_map_info;
    grid_map_type m_grid_map;
// [private members]

public:

    assembly_base( uint_t d1, uint_t d2, uint_t d3 ):
        m_d1(d1)
        , m_d2(d2)
        , m_d3(d3)
        , m_grid_info(d1, d2, d3, geo_map::basisCardinality, 3)
        , m_grid(m_grid_info, 0., "grid")
        , m_grid_map_info(d1, d2, d3, geo_map::basisCardinality)
        , m_grid_map(m_grid_map_info, 0., "grid_map")
        {        }

    grid_type const& get_grid() const {return m_grid;}
    grid_type& grid() {return m_grid;}
    grid_map_type const& get_grid_map() const {return m_grid_map;}
    grid_map_type& grid_map() {return m_grid_map;}

}; //struct assembly

//struct definition
template < typename ... Types >
struct domain_type_tuple;

/**
    default case: just forwarding the args
    necessary in order to allow arbitrary order of the template arguments
 */
template <>
struct domain_type_tuple<>{

    static const ushort_t size=0;

    template <typename ... MPLList>
    gridtools::domain_type< boost::mpl::vector<MPLList ...> >
    domain(typename MPLList::storage_type& ...  storages_ ){
        return domain_type<boost::mpl::vector< MPLList ...> >(boost::fusion::make_vector(&storages_ ...));
    }

};

template < typename Geometry, typename ... Rest >
struct domain_type_tuple<assembly_base<Geometry>, Rest ... > : domain_type_tuple<Rest ...> {

private:
    using super = domain_type_tuple< Rest ...>;
    using as_t = assembly_base<Geometry>;
    as_t & m_as;

public:

    domain_type_tuple(as_t & as_) : m_as(as_) {}

    /**I have to define here the placeholders to the storages used: the temporary storages get internally managed, while
       non-temporary ones must be instantiated by the user. In this example all the storages are non-temporaries.*/

    typedef arg<0, typename as_t::grid_type >       p_grid_points;
    static const ushort_t size=1;

    template <typename ... MPLList>
    gridtools::domain_type< boost::mpl::vector<p_grid_points
                                               , typename boost::remove_reference
                                               <typename boost::remove_pointer<
                                                    MPLList>::type>::type ...> >
    domain(typename boost::remove_reference
               <typename boost::remove_pointer<
               typename MPLList::storage_type>::type>::type& ...  storages_ ){
        return super::template domain<p_grid_points
                                      , typename boost::remove_reference
                                      <typename boost::remove_pointer<
                                           MPLList>::type>::type ...
                                      >
            (m_as.grid(), storages_ ...);
    }

};

#endif //CXX11_ENABLED

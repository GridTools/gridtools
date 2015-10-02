#pragma once

// [includes]
#include "assembly_base.hpp"
#include "assembly_functors.hpp"
// [includes]
#ifdef CXX11_ENABLED

// [namespaces]
using namespace gridtools;
using namespace enumtype;
using namespace expressions;
// [namespaces]

typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
typedef gridtools::interval<gridtools::level<0,-2>, gridtools::level<1,1> > axis;

// [storage_types]
template <typename ... Geometry>
struct assembly{};


template <typename Geometry>
struct assembly<Geometry> : public assembly_base<Geometry> {

    using super=assembly_base<Geometry>;
    using cub=typename Geometry::cub;
    using geo_map=typename Geometry::geo_map;

    //                      dims  x y z  qp
    //                   strides  1 x xy xyz
    using storage_type_info=storage_info<gridtools::layout_map<0,1,2,3>, __COUNTER__ >;
    using jacobian_type_info=storage_info<gridtools::layout_map<0,1,2,3,4,5>, __COUNTER__ >;

    using storage_type=storage_t< storage_type_info >;
    using jacobian_type=storage_t< jacobian_type_info >;
    static const int_t edge_points=geo_map::hypercube_t::template boundary_w_dim<1>::n_points::value;
// [storage_types]


    /**I have to define here the placeholders to the storages used: the temporary storages get internally managed, while
       non-temporary ones must be instantiated by the user. In this example all the storages are non-temporaries.*/
    typedef arg<super::size+0, jacobian_type >   p_jac;
    typedef arg<super::size+1, typename Geometry::weights_storage_t >   p_weights;
    typedef arg<super::size+2, storage_type >    p_jac_det;
    typedef arg<super::size+3, jacobian_type >   p_jac_inv;
    static const uint_t size=super::size+4;

// [private members]
    uint_t m_d1, m_d2, m_d3;
private:

    Geometry & m_fe_backend;
    jacobian_type_info m_jac_info;
    storage_type_info m_jac_det_info;

    jacobian_type m_jac;
    storage_type m_jac_det;
    jacobian_type m_jac_inv;

public:

    /**
       @brief construct the basic storages involved in the finite elements assembly

       NOTE: fe_backend_ contains all the storages which are local to one element, while this class instantiates
       the expensive storages, spanning the whole computational domain.
     */
    assembly(Geometry& fe_backend_, uint_t d1, uint_t d2, uint_t d3 ):
        super(d1, d2, d3)
        , m_fe_backend(fe_backend_)
        , m_d1(d1)
        , m_d2(d2)
        , m_d3(d3)
        , m_jac_info(d1, d2, d3, cub::numCubPoints(), 3, 3)
        , m_jac_det_info(d1, d2, d3, cub::numCubPoints())
        , m_jac(m_jac_info, "jacobian")
        , m_jac_det(m_jac_det_info, "jacobian det")
        , m_jac_inv(m_jac_info, "jacobian inv")
        {        }

    jacobian_type const& get_jac() const {return m_jac;}
    storage_type const& get_jac_det() const {return m_jac_det;}
    jacobian_type const& get_jac_inv() const {return m_jac_inv;}
    typename Geometry::weights_storage_t const& get_cub_weights() const {return m_fe_backend.cub_weights();}
    jacobian_type & jac() {return m_jac;}
    storage_type & jac_det() {return m_jac_det;}
    jacobian_type & jac_inv() {return m_jac_inv;}
    typename Geometry::weights_storage_t & cub_weights() {return m_fe_backend.cub_weights();}

    /**
       @brief adds few extra placeholders<->storages items to the domain_type
     */
    template <typename ... MPLList>
    auto domain( typename MPLList::storage_type& ...  storages_
        )
        -> decltype(this->template domain_base< p_jac,
                    p_weights, p_jac_det, p_jac_inv,
                    MPLList ...>
                    ( m_jac, m_fe_backend.cub_weights(), m_jac_det, m_jac_inv , storages_ ...))
        {
            return this->template domain_base< p_jac,
                                               p_weights, p_jac_det, p_jac_inv,
                                               MPLList ...
                                               >
                ( m_jac, m_fe_backend.cub_weights(), m_jac_det, m_jac_inv , storages_ ...);
        }

// [private members]

}; //struct assembly

template<typename GEO>
const int_t assembly< GEO >::edge_points;



#endif //CXX11_ENABLED

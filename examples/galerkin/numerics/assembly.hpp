#pragma once

// [includes]
#include "assembly_base.hpp"
#include "../functors/assembly_functors.hpp"
#include "../functors/mass.hpp"
#include "../functors/advection.hpp"
#include "../functors/stiffness.hpp"
// [includes]

// [namespaces]
// using namespace gridtools;
// using namespace enumtype;
// using namespace expressions;
// [namespaces]
namespace gdl{

typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
typedef gridtools::interval<gridtools::level<0,-2>, gridtools::level<1,1> > axis;

template <typename Geometry>
struct assembly  {

    using geometry_t=Geometry;
    using super=assembly_base<Geometry>;
    using cub=typename Geometry::cub;
    using geo_map=typename Geometry::geo_map;
    using weights_storage_t = typename Geometry::weights_storage_t;
    using phi_t = typename geometry_t::basis_function_storage_t;
    using dphi_t = typename geometry_t::grad_storage_t;
    //                      dims  x y z  qp
    //                   strides  1 x xy xyz
    using storage_type_info=storage_info<__COUNTER__, layout_tt<4> >;
    using jacobian_type_info=storage_info<__COUNTER__, layout_tt<6> >;

    using storage_type=storage_t< storage_type_info >;
    using jacobian_type=storage_t< jacobian_type_info >;
    static const int_t edge_points=geo_map::hypercube_t::template boundary_w_dim<1>::n_points::value;
// [storage_types]


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
        m_fe_backend(fe_backend_)
        , m_d1(d1)
        , m_d2(d2)
        , m_d3(d3)
        , m_jac_info(d1, d2, d3, cub::numCubPoints(), 3, 3)
        , m_jac_det_info(d1, d2, d3, cub::numCubPoints())
        , m_jac(m_jac_info, 0., "jacobian")
        , m_jac_det(m_jac_det_info, 0., "jacobian det")
        , m_jac_inv(m_jac_info, 0., "jacobian inv")
        {
        }

    jacobian_type const& get_jac() const {return m_jac;}
    storage_type const& get_jac_det() const {return m_jac_det;}
    jacobian_type const& get_jac_inv() const {return m_jac_inv;}
    typename Geometry::weights_storage_t const& get_cub_weights() const {return m_fe_backend.cub_weights();}
    Geometry const& get_fe_backend() const {return m_fe_backend;}

    Geometry & fe_backend() {return m_fe_backend;}
    jacobian_type & jac() {return m_jac;}
    storage_type & jac_det() {return m_jac_det;}
    jacobian_type & jac_inv() {return m_jac_inv;}
    weights_storage_t & cub_weights() {return m_fe_backend.cub_weights();}

    // [private members]

}; //struct assembly

template<typename GEO>
const int_t assembly< GEO >::edge_points;


template <typename Geometry, typename ... Rest>
struct aggregator_type_tuple<assembly<Geometry>,  Rest ... > : aggregator_type_tuple< Rest ...> {

private:
    using super = aggregator_type_tuple< Rest ...>;
    using as_t = assembly<Geometry>;
    as_t & m_as;

public:

    template <typename ... Args>
    aggregator_type_tuple(as_t & as_, Args & ... args_) : super(args_ ...), m_as(as_) {}

    /**I have to define here the placeholders to the storages used: the temporary storages get internally managed, while
       non-temporary ones must be instantiated by the user. In this example all the storages are non-temporaries.*/
    typedef gt::arg<super::size+0, typename as_t::jacobian_type >   p_jac;
    typedef gt::arg<super::size+1, typename as_t::geometry_t::weights_storage_t >   p_weights;
    typedef gt::arg<super::size+2, typename as_t::storage_type >    p_jac_det;
    typedef gt::arg<super::size+3, typename as_t::jacobian_type >   p_jac_inv;
    typedef gt::arg<super::size+4, typename as_t::geometry_t::basis_function_storage_t> p_phi;
    typedef gt::arg<super::size+5, typename as_t::geometry_t::grad_storage_t> p_dphi;
    static const uint_t size=super::size+6;

    /**
       @brief adds few extra placeholders<->storages items to the aggregator_type
     */
    template <typename ... MPLList>
    auto domain( typename boost::remove_reference
                <typename boost::remove_pointer<
                 typename MPLList::storage_type>::type>::type& ...  storages_
        )
        -> decltype(super::template domain<
                     p_jac
                    , p_weights
                    , p_jac_det
                    , p_jac_inv
                    , p_phi
                    , p_dphi
                    , typename boost::remove_reference
                    <typename boost::remove_pointer<
                    MPLList>::type>::type ...
                    >
                    ( m_as.jac()
                      , m_as.fe_backend().cub_weights()
                      , m_as.jac_det()
                      , m_as.jac_inv()
                      , m_as.fe_backend().val()
                      , m_as.fe_backend().grad()
                      , storages_ ...))
        {
            return super::template domain<    p_jac
                                              , p_weights
                                              , p_jac_det
                                              , p_jac_inv
                                              , p_phi
                                              , p_dphi
                                              , typename boost::remove_reference
                                              <typename boost::remove_pointer<
                                                   MPLList>::type>::type ...
                                              >
                ( m_as.jac()
                  , m_as.fe_backend().cub_weights()
                  , m_as.jac_det()
                  , m_as.jac_inv()
                  , m_as.fe_backend().val()
                  , m_as.fe_backend().grad()
                  , storages_ ...);
        }


    template<enumtype::Shape S>
    struct update_jac{
        auto static esf() ->
            decltype(gt::make_stage<functors::update_jac<typename as_t::geometry_t , S> >(typename super::p_grid_points(), p_dphi(), p_jac()))
        {
            return gt::make_stage<functors::update_jac<typename as_t::geometry_t , S> >(typename super::p_grid_points(), p_dphi(), p_jac());
        }
    };

    //TODO: generalize the foillowing functors
    template <typename FE, typename Cubature>
    struct mass{
        template <typename Phi, typename Mass>
        auto static esf(Phi, Mass) ->
            decltype(gt::make_stage<functors::mass >(p_jac_det(), p_weights(), Phi(), Phi(), Mass()))
        {
            return gt::make_stage<functors::mass >(p_jac_det(), p_weights(), Phi(), Phi(), Mass());
        }
    };

    template <typename FE, typename Cubature>
    struct stiffness{
        template<typename DPhi, typename Stiff>
        auto static esf(DPhi, Stiff) ->
            decltype(gt::make_stage<functors::stiffness<FE , Cubature> >(p_jac_det(), p_jac_inv(), p_weights(), DPhi(), DPhi(), Stiff()))
        {
            return gt::make_stage<functors::stiffness<FE , Cubature> >(typename super::p_jac_det(), p_jac_inv(), p_weights(), DPhi(), DPhi(), Stiff());
        }
    };

    template <typename FE, typename Cubature>
    struct advection{
        template<typename Beta, typename Phi, typename DPhi, typename Adv>
        auto static esf(Beta, Phi, DPhi, Adv) ->
            decltype(gt::make_stage<functors::advection<FE , Cubature> >(p_jac_det(), p_jac_inv(), p_weights(), Beta(), DPhi(), Phi(), Adv()))
        {
            //TODO check that the inverse is computed
            return gt::make_stage<functors::advection<FE , Cubature> >(p_jac_det(), p_jac_inv(), p_weights(), Beta(), DPhi(), Phi(), Adv());
        }
    };

};

}//namespace gdl

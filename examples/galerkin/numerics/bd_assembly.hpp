#pragma once

/**
   @file
   @brief Definition of the quantities needed for performing computations on the boundary
*/

// [includes]
#include "assembly.hpp"
#include "../functors/bd_assembly_functors.hpp"
#include "../functors/dg_fluxes.hpp"

namespace gdl{
/**
   @brief Definition of the quantities needed for performing computations on the boundary

   The following class holds the discretization containers spanning the whole iteration domain, as opposed to
   \ref intrepid.hpp which holds the local elemental quantities.
   NOTE: underlying 3D assumption

*/
template <typename Boundary>
struct bd_assembly {

    // static const int_t n_faces=geo_map::hypercube_t::template n_boundary_w_dim<Boundary::spaceDim>::value;
    using boundary_t = Boundary;
    using bd_cub=typename Boundary::cub;
    // using super = assembly_base<Geometry>;

    using face_normals_type_info=storage_info< __COUNTER__, layout_tt<3,4,5>>;
    using face_normals_type=storage_t< face_normals_type_info >;
    using storage_type_info=storage_info< __COUNTER__, layout_tt<3,4> >;
    using storage_type=storage_t< storage_type_info >;
    using jacobian_type_info=storage_info<__COUNTER__, layout_tt<3,4,5,6> >;
    using jacobian_type=storage_t< jacobian_type_info >;
    using bd_vector_storage_info_t=storage_info< __COUNTER__, layout_tt<3> >;//TODO change: iterate on faces
    using bd_vector_type=storage_t< bd_vector_storage_info_t >;

private:
    boundary_t & m_bd_backend;
    jacobian_type_info m_jac_info;
    face_normals_type_info m_normals_info;
    storage_type_info m_bd_measure_info;
    /**overdimensioned. Reduce*/

    jacobian_type m_bd_jac;
    face_normals_type m_normals;
    storage_type m_bd_measure;

public:

    Boundary & bd_backend()  { return m_bd_backend;}
    jacobian_type & bd_jac()  {return m_bd_jac;}
    face_normals_type & normals()  {return m_normals;}
    storage_type & bd_measure()  { return m_bd_measure;}

    jacobian_type const& get_bd_jac() const {return m_bd_jac;}
    face_normals_type const& get_normals() const {return m_normals;}
    storage_type const& get_bd_measure() const { return m_bd_measure;}


    typename Boundary::tangent_storage_t & ref_normals() const {return m_bd_backend.ref_normals();}
    typename Boundary::tangent_storage_t const& get_ref_normals() const {return m_bd_backend.ref_normals();}

    bd_assembly( Boundary& bd_backend_,
             // Geometry& fe_backend_,
              uint_t d1, uint_t d2, uint_t d3) :
        m_bd_backend(bd_backend_)
        , m_jac_info(d1, d2, d3, bd_cub::numCubPoints(), 3, 3, bd_backend_.n_boundaries())
        , m_normals_info(d1, d2, d3, bd_cub::numCubPoints(), 3, bd_backend_.n_boundaries())
        , m_bd_measure_info(d1, d2, d3, bd_cub::numCubPoints(), bd_backend_.n_boundaries())
        , m_bd_jac(m_jac_info, 0., "bd jac")
        , m_normals(m_normals_info, 0., "normals")
        , m_bd_measure(m_bd_measure_info, 0., "bd measure")
    {}

};


template <typename Boundary, typename ... Rest>
struct aggregator_type_tuple<bd_assembly<Boundary>,  Rest ... > : aggregator_type_tuple< Rest ...> {

private:
    using super = aggregator_type_tuple< Rest ...>;
    using as_t = bd_assembly<Boundary>;
    as_t & m_as;

public:
    template<typename ... Args>
    aggregator_type_tuple(as_t & as_, Args & ... args_) : super(args_ ...), m_as(as_) {}

    typedef gt::arg<super::size+0, typename as_t::jacobian_type >       p_bd_jac;
    typedef gt::arg<super::size+1, typename as_t::face_normals_type >                   p_normals;
    typedef gt::arg<super::size+2, typename as_t::storage_type >        p_bd_measure;
    typedef gt::arg<super::size+3, typename as_t::boundary_t::weights_storage_t> p_bd_weights;
    typedef gt::arg<super::size+4, typename as_t::boundary_t::tangent_storage_t> p_ref_normals;
    typedef gt::arg<super::size+6, typename as_t::boundary_t::basis_function_storage_t> p_bd_phi;
    typedef gt::arg<super::size+7, typename as_t::boundary_t::grad_storage_t> p_bd_dphi;
    static const ushort_t size=super::size+8;

    // template <typename ... MPLList>
    // int domain( typename MPLList::storage_type & ...  storages_ )
    // {
    //     typedef typename boost::mpl::vector<MPLList ...>::fuck fuck;
    //     return 0;
    // }


    /**
       @brief adds few extra placeholders<->storages items to the aggregator_type
    */
    template <typename ... MPLList>
    auto domain(typename boost::remove_reference
                <typename boost::remove_pointer<
                typename MPLList::storage_type>::type>::type & ...  storages_ )
        -> decltype(super::template domain<
                    p_bd_jac
                    , p_normals
                    , p_bd_measure
                    , p_bd_weights
                    , p_ref_normals
                    , p_bd_phi
                    , p_bd_dphi
                    , typename boost::remove_reference
                    <typename boost::remove_pointer<
                    MPLList>::type>::type ...
                    >( m_as.bd_jac()
                       , m_as.normals()
                       , m_as.bd_measure()
                       , m_as.bd_backend().bd_cub_weights()
                       , m_as.bd_backend().ref_normals()
                       , m_as.bd_backend().val()
                       , m_as.bd_backend().grad()
                       , storages_ ...
                        ))
    {
        return super::template domain<  p_bd_jac
                                        , p_normals
                                        , p_bd_measure
                                        , p_bd_weights
                                        , p_ref_normals
                                        , p_bd_phi
                                        , p_bd_dphi
                                        , typename boost::remove_reference
                                        <typename boost::remove_pointer<
                                             MPLList>::type>::type ...
                                        >
            ( m_as.bd_jac()
              , m_as.normals()
              , m_as.bd_measure()
              , m_as.bd_backend().bd_cub_weights()
              , m_as.bd_backend().ref_normals()
              , m_as.bd_backend().val()
              , m_as.bd_backend().grad()
              , storages_ ...
                );
    }

    struct compute_face_normals{
        // template<typename DPhi, typename Stiff>
        auto static esf() ->
            decltype(gt::make_stage<functors::compute_face_normals<typename as_t::boundary_t> >(p_bd_jac(), p_ref_normals(), p_normals()))
        {
            return gt::make_stage<functors::compute_face_normals<typename as_t::boundary_t> >(p_bd_jac(), p_ref_normals(), p_normals());
        }
    };


    struct bd_integrate{
        template<typename In, typename Out>
        auto static esf(In, Out) ->
            decltype(gt::make_stage<functors::bd_integrate<typename as_t::boundary_t> >(p_bd_phi(), p_bd_measure(), p_bd_weights(), In(), Out()))
        {
            return gt::make_stage<functors::bd_integrate<typename as_t::boundary_t> >(p_bd_phi(), p_bd_measure(), p_bd_weights(), In(), Out());
        }
    };



    template<enumtype::Shape S>
    struct update_bd_jac{
        auto static esf() ->
            decltype(gt::make_stage<functors::update_bd_jac<typename as_t::boundary_t , S> >(typename super::p_grid_points(), p_bd_dphi(), p_bd_jac()))
        {
            return gt::make_stage<functors::update_bd_jac<typename as_t::boundary_t , S> >(typename super::p_grid_points(), p_bd_dphi(), p_bd_jac());
        }
    };

    template<ushort_t Codimension>
    struct measure{
        auto static esf() ->
            decltype(gt::make_stage<functors::measure<typename as_t::boundary_t , Codimension> >(p_bd_jac(), p_bd_measure()))
        {
            return gt::make_stage<functors::measure<typename as_t::boundary_t, Codimension> >(p_bd_jac(),  p_bd_measure());
        }
    };

    template<typename Flux >
    struct lax_friedrich {

        template<typename Sol, typename Result>
        auto static esf(Sol, Result) ->
            decltype(gt::make_stage<functors::lax_friedrich<typename as_t::boundary_t, Flux> >(Sol(), Result()))
        {
            return gt::make_stage<functors::lax_friedrich<typename as_t::boundary_t, Flux> >(Sol(), Result()); //mass
        }
    };

    template<typename Flux >
    struct upwind {

        template<typename Sol, typename Beta, typename Result>
        auto static esf(Sol, Beta, Result) ->
            decltype(gt::make_stage<functors::upwind >(as_t::p_normals(), Sol(), Beta(), Result()))
        {
            return gt::make_stage<functors::upwind >(as_t::p_normals(), Sol(), Beta(), Result()); //mass
        }
    };

};
}//namespace gdl

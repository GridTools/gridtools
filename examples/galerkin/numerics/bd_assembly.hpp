#pragma once
/**
   @file
   @brief Definition of the quantities needed for performing computations on the boundary
*/

// [includes]
#include "assembly.hpp"
#include "bd_assembly_functors.hpp"

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

    using face_normals_type_info=storage_info< layout_tt<3,4,5>, __COUNTER__ >;
    using face_normals_type=storage_t< face_normals_type_info >;
    using storage_type_info=storage_info< layout_tt<3,4>, __COUNTER__ >;
    using storage_type=storage_t< storage_type_info >;
    using jacobian_type_info=storage_info<layout_tt<3,4,5,6>, __COUNTER__ >;
    using jacobian_type=storage_t< jacobian_type_info >;

private:
    boundary_t & m_bd_backend;
    jacobian_type_info m_jac_info;
    face_normals_type_info m_normals_info;
    storage_type_info m_bd_measure_info;

    jacobian_type m_bd_jac;
    face_normals_type m_normals;
    storage_type m_bd_measure;

public:

    jacobian_type & bd_jac()  {return m_bd_jac;}
    face_normals_type & normals()  {return m_normals;}
    storage_type & bd_measure()  { return m_bd_measure;}
    Boundary & bd_backend()  { return m_bd_backend;}

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
struct domain_type_tuple<bd_assembly<Boundary>,  Rest ... > : domain_type_tuple< Rest ...> {

private:
    using super = domain_type_tuple< Rest ...>;
    using as_t = bd_assembly<Boundary>;
    as_t & m_as;

public:
    template<typename ... Args>
    domain_type_tuple(as_t & as_, Args & ... args_) : super(args_ ...), m_as(as_) {}

    typedef arg<super::size+0, typename as_t::jacobian_type >       p_bd_jac;
    typedef arg<super::size+1, typename as_t::face_normals_type >                   p_normals;
    typedef arg<super::size+2, typename as_t::storage_type >        p_bd_measure;
    typedef arg<super::size+3, typename as_t::boundary_t::weights_storage_t> p_bd_weights;
    typedef arg<super::size+4, typename as_t::boundary_t::tangent_storage_t> p_ref_normals;
    static const ushort_t size=super::size+5;


    /**
       @brief adds few extra placeholders<->storages items to the domain_type
     */
    template <typename ... MPLList>
    auto domain(typename MPLList::storage_type& ...  storages_ )
        -> decltype(super::template domain<
                    p_bd_jac
                    , p_normals
                    , p_bd_measure
                    , p_bd_weights
                    , p_ref_normals
                    , MPLList ...
                    >( m_as.bd_jac()
                       ,  m_as.normals()
                       , m_as.bd_measure()
                       , m_as.bd_backend().bd_cub_weights()
                       , m_as.bd_backend().ref_normals()
                       , storages_ ...
                        ))
        {
            return super::template domain<  p_bd_jac
                                               , p_normals
                                               , p_bd_measure
                                               , p_bd_weights
                                               , p_ref_normals
                                               ,MPLList ...
                                               >
                ( m_as.bd_jac()
                  , m_as.normals()
                  , m_as.bd_measure()
                  , m_as.bd_backend().bd_cub_weights()
                  , m_as.bd_backend().ref_normals()
                  , storages_ ...
                    );
        }

};

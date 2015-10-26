#pragma once
// [includes]
#include "assembly.hpp"
#include "bd_assembly_functors.hpp"

template <typename Boundary, typename Geometry>
struct assembly<Boundary, Geometry > : public assembly_base<Geometry> {

    // static const int_t n_faces=geo_map::hypercube_t::template n_boundary_w_dim<Boundary::spaceDim>::value;
    using bd_cub=typename Boundary::cub;
    using super = assembly_base<Geometry>;

    using face_normals_type_info=storage_info< layout_tt<3,4>, __COUNTER__ >;
    using face_normals_type=storage_t< face_normals_type_info >;
    using storage_type_info=storage_info< layout_tt<3>, __COUNTER__ >;
    using storage_type=storage_t< storage_type_info >;
    using jacobian_type_info=storage_info<layout_tt<3,4,5>, __COUNTER__ >;
    using jacobian_type=storage_t< jacobian_type_info >;

    typedef arg<super::size+0, jacobian_type >       p_bd_jac;
    typedef arg<super::size+1, jacobian_type >       p_projected_jac;
    typedef arg<super::size+2, face_normals_type >                   p_normals;
    typedef arg<super::size+3, storage_type >        p_bd_measure;
    typedef arg<super::size+4, typename Boundary::weights_storage_t> p_bd_weights;
    typedef arg<super::size+5, typename Boundary::tangent_storage_t> p_ref_normals;
    static const ushort_t size=super::size+6;

private:
    jacobian_type_info m_jac_info;
    face_normals_type_info m_normals_info;
    storage_type_info m_bd_measure_info;

    jacobian_type m_bd_jac;
    jacobian_type m_projected_jac;
    face_normals_type m_normals;
    storage_type m_bd_measure;
    Boundary & m_bd_backend;

public:

    jacobian_type const& get_jac() const {return m_bd_jac;}


    face_normals_type const& get_normals() const {return m_normals;}

    typename Boundary::tangent_storage_t const& get_ref_normals() const {return m_bd_backend.ref_normals();}

    assembly( Boundary& bd_backend_,
             // Geometry& fe_backend_,
              uint_t d1, uint_t d2, uint_t d3) :
        super( d1, d2, d3)
        , m_jac_info(d1, d2, d3, bd_cub::numCubPoints(), 3, 3)
        , m_normals_info(d1, d2, d3, bd_cub::numCubPoints(), 3)
        , m_bd_measure_info(d1, d2, d3, bd_cub::numCubPoints())
        , m_bd_jac(m_jac_info, 0., "bd jac")
        , m_projected_jac(m_jac_info, 0., "projected jac")
        , m_normals(m_normals_info, 0., "normals")
        , m_bd_measure(m_bd_measure_info, 0., "bd measure")
        , m_bd_backend(bd_backend_)
        {}



    /**
       @brief adds few extra placeholders<->storages items to the domain_type
     */
    template <typename ... MPLList>
    auto domain(typename MPLList::storage_type& ...  storages_ )
        -> decltype(this->template domain_base< p_bd_jac
                    , p_projected_jac
                    , p_normals
                    , p_bd_measure
                    , p_bd_weights
                    , p_ref_normals
                    , MPLList ...
                    >( m_bd_jac
                       , m_projected_jac
                       ,  m_normals
                       , m_bd_measure
                       , m_bd_backend.bd_cub_weights()
                       , m_bd_backend.m_ref_normals
                       , storages_ ...
                        ))
        {
            return this->template domain_base< p_bd_jac
                                               , p_projected_jac
                                               , p_normals
                                               , p_bd_measure
                                               , p_bd_weights
                                               , p_ref_normals
                                               ,MPLList ...
                                               >
                ( m_bd_jac
                  , m_projected_jac
                  , m_normals
                  , m_bd_measure
                  , m_bd_backend.bd_cub_weights()
                  , m_bd_backend.m_ref_normals
                  , storages_ ...
                    );
        }

};

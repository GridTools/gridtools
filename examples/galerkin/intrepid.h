#pragma once

// [includes]
#include <Intrepid_FunctionSpaceTools.hpp>
#include <Intrepid_Types.hpp>

#include <gridtools.hpp>
#include <stencil-composition/accessor.hpp>
#include <stencil-composition/interval.hpp>

#include <boost/type_traits.hpp>
#include "basis_functions.h"
#include "cubature.h"
// [includes]

#define REORDER

// #include "assembly.h"

namespace gridtools{
namespace intrepid{

    template <typename FE, typename Cub>
    struct discretization{
        using fe=FE;
        using cub=Cub;
        static const enumtype::Shape parent_shape=FE::shape;
        static const enumtype::Shape bd_shape=shape_property<FE::shape>::boundary;

        // [test]
        typedef storage_t<layout_map<0,1,2> >::value_type value_type;
        GRIDTOOLS_STATIC_ASSERT(fe::layout_t::template at_<0>::value < 3 && fe::layout_t::template at_<1>::value < 3 && fe::layout_t::template at_<2>::value < 3,
                                "the first three numbers in the layout_map must be a permutation of {0,1,2}. ");
        typedef storage_t<layout_map<0,1,2> > weights_storage_t;
        typedef storage_t<layout_map<0,1,2> > grad_storage_t;
        typedef storage_t<layout_map<0,1,2> > basis_function_storage_t;

        storage_t<layout_map<0,1,2> > m_cub_points_s;
        weights_storage_t m_cub_weights_s;
        std::unique_ptr<grad_storage_t> m_grad_at_cub_points_s;
        std::unique_ptr<basis_function_storage_t> m_phi_at_cub_points_s;

        storage_t<layout_map<0,1,2> > // const
        & cub_points()// const
            {return m_cub_points_s;}
        weights_storage_t // const
        & cub_weights()// const
            {return m_cub_weights_s;}
        grad_storage_t // const
        & local_gradient()// const
            {return *m_grad_at_cub_points_s;}

        basis_function_storage_t // const
        & basis_function()// const
            {return *m_phi_at_cub_points_s;}

        discretization() :
            m_cub_points_s(cub::numCubPoints, fe::spaceDim,1)
            , m_cub_weights_s(cub::numCubPoints,1,1)
            , m_grad_at_cub_points_s()
            , m_phi_at_cub_points_s()
            {
            }

        void compute(Intrepid::EOperator const& operator_){

                // storage_t<layout_map<0,1,2> > cub_points_i(m_cub_points_s, 2);
            Intrepid::FieldContainer<double> cub_points_i(cub::numCubPoints, fe::spaceDim);

                // storage_t<layout_map<0,1,2> > cub_weights_i(m_cub_weights_s, 1);
            Intrepid::FieldContainer<double> cub_weights_i(cub::numCubPoints);

                //copy the values
                for (uint_t q=0; q<cub::numCubPoints; ++q)
                {
                    m_cub_weights_s(q,0,0)=cub_weights_i(q);
                    for (uint_t j=0; j<fe::spaceDim; ++j)
                        m_cub_points_s(q,j,0)=cub_points_i(q,j);
                }

                // storage_t<layout_map<0,1,2> > grad_at_cub_points_i(m_grad_at_cub_points_s);
                Intrepid::FieldContainer<double> grad_at_cub_points_i(fe::basisCardinality, cub::numCubPoints, fe::spaceDim);

                // retrieve cub points and weights
                cub::cub->getCubature(cub_points_i, cub_weights_i);

                switch (operator_){
                case Intrepid::OPERATOR_GRAD :
                {
                    m_grad_at_cub_points_s=std::unique_ptr<grad_storage_t>(new grad_storage_t(fe::basisCardinality, cub::numCubPoints, fe::spaceDim));

                    // evaluate grad operator at cub points
                    fe::hexBasis.getValues(grad_at_cub_points_i, cub_points_i, Intrepid::OPERATOR_GRAD);

                    for (uint_t q=0; q<cub::numCubPoints; ++q)
                        for (uint_t j=0; j<fe::spaceDim; ++j)
                            for (uint_t i=0; i<fe::basisCardinality; ++i)
                                for (uint_t j=0; j<fe::spaceDim; ++j)
				    (*m_grad_at_cub_points_s)(i,q,j)=grad_at_cub_points_i(i,q,j);
                    break;
                }
                case Intrepid::OPERATOR_VALUE :
                {
                    m_phi_at_cub_points_s=std::unique_ptr<basis_function_storage_t>(new basis_function_storage_t(fe::basisCardinality, cub::numCubPoints, 1));
                    Intrepid::FieldContainer<double> phi_at_cub_points_i(fe::basisCardinality, cub::numCubPoints);

                    fe::hexBasis.getValues(phi_at_cub_points_i, cub_points_i, Intrepid::OPERATOR_VALUE);
                    //copy the values
                    for (uint_t q=0; q<cub::numCubPoints; ++q)
                        // for (uint_t j=0; j<fe::spaceDim; ++j)
			for (uint_t i=0; i<fe::basisCardinality; ++i){
			    (*m_phi_at_cub_points_s)(i,q,0)=phi_at_cub_points_i(i,q);
			}
                    break;
                }
                default : assert(false);
                }

        }
    };


    template < typename GeoMap, typename Cubature >
    struct geometry : public discretization<GeoMap, Cubature>{

        using geo_map=GeoMap;
        using super=discretization<GeoMap, Cubature>;

        using bd_geo_map = reference_element<geo_map::order
                                             , geo_map::basis
                                             , shape_property<geo_map::shape>::boundary>;

        storage_t<layout_map<0,1,2> > m_local_grid_s;
#ifdef REORDER
        storage_t<layout_map<0,1,2> > m_local_grid_reordered_s;
#endif

        geometry() :
            //create the local grid
            m_local_grid_s(geo_map::basisCardinality, geo_map::spaceDim,1)
#ifdef REORDER
            , m_local_grid_reordered_s(geo_map::basisCardinality, geo_map::spaceDim,1)
#endif
            {
                // storage_t<layout_map<0,1,2> > local_grid_i(m_local_grid_s, 2);
                Intrepid::FieldContainer<double> local_grid_i(geo_map::basisCardinality, geo_map::spaceDim);
                geo_map::hexBasis.getDofCoords(local_grid_i);
                for (uint_t i=0; i<geo_map::basisCardinality; ++i)
                    for (uint_t j=0; j<geo_map::spaceDim; ++j)
                        m_local_grid_s(i,j,0)=local_grid_i(i,j);

                //! [reorder]
                std::vector<uint_t> permutations( geo_map::basisCardinality );
                std::vector<uint_t> to_reorder( geo_map::basisCardinality );
                //sorting the a vector containing the point coordinates with priority i->j->k, and saving the permutation
#ifdef REORDER
                // fill in the reorder vector such that the larger numbers correspond to larger strides
                for(uint_t i=0; i<geo_map::basisCardinality; ++i){
                    to_reorder[i]=(m_local_grid_s(i,geo_map::layout_t::template at_<0>::value)+2)*4+(m_local_grid_s(i,geo_map::layout_t::template at_<1>::value)+2)*2+(m_local_grid_s(i,geo_map::layout_t::template at_<2>::value)+2);
                    permutations[i]=i;
                }

                std::sort(permutations.begin(), permutations.end(),
                  [&to_reorder](uint_t a, uint_t b){
                      return to_reorder[a]<to_reorder[b];
                  } );

                // storage_t<layout_map<0,1,2> >::storage_t  local_grid_reordered_s(geo_map::basisCardinality, geo_map::spaceDim,1);
                // storage_t<layout_map<0,1,2> >  local_grid_reordered_i(m_local_grid_reordered_s, 2);
                uint_t D=geo_map::basisCardinality;

                //applying the permutation to the grid
                for(uint_t i=0; i<D; ++i){//few redundant loops
                    {
                        m_local_grid_reordered_s(i, 0, 0)=m_local_grid_s(permutations[i],0,0);
                        m_local_grid_reordered_s(i, 1, 0)=m_local_grid_s(permutations[i],1,0);
                        m_local_grid_reordered_s(i, 2, 0)=m_local_grid_s(permutations[i],2,0);
                    }
                }
                //! [reorder]
#endif

                super::compute(Intrepid::OPERATOR_GRAD);
            }

        storage_t<layout_map<0,1,2> > const& grid(){return m_local_grid_s;}

    };


    template<typename GeoMap, ushort_t Order>
    class boundary_cub // : public Base
    {
    public:
        typedef storage_t<layout_map<0,1,2> > weights_storage_t;
        typedef storage_t<layout_map<0,1,2> > points_storage_t;
        // typedef storage_t<layout_map<0,1,2> > points_lifted_storage_t;

        using geo_map = GeoMap;
        using value_t=float_type;
        static const enumtype::Shape parent_shape=geo_map::shape;
        static const enumtype::Shape bd_shape=shape_property<geo_map::shape>::boundary;
        static const ushort_t n_sub_cells=shape_property<geo_map::shape>::n_sub_cells;
        using bd_cub = cubature<Order, bd_shape>;

        //private:

        points_storage_t m_bd_cub_pts;
        weights_storage_t m_bd_cub_weights;
        // points_lifted_storage_t  m_bd_cub_pts_lifted;

    public:
        boundary_cub()
            :
            m_bd_cub_pts(bd_cub::numCubPoints, geo_map::spaceDim-1, 1)
            , m_bd_cub_weights(bd_cub::numCubPoints, 1, 1)
            // , m_bd_cub_pts_lifted(bd_cub::numCubPoints, geo_map::spaceDim, 1)
            {
                Intrepid::FieldContainer<value_t> bd_cub_pts_(bd_cub::numCubPoints, geo_map::spaceDim-1);
                Intrepid::FieldContainer<value_t> bd_cub_weights_(bd_cub::numCubPoints);

                bd_cub::cub->getCubature(bd_cub_pts_, bd_cub_weights_);

                for (uint_t i=0; i<bd_cub::numCubPoints; ++i){
                    m_bd_cub_weights(i,0,0)=bd_cub_weights_(i);
                    for (uint_t j=0; j<geo_map::spaceDim-1; ++j){
                        m_bd_cub_pts(i,j,0)=bd_cub_pts_(i,j);
                    }
                }
            }

        /**@brief performs the lift of the reference subcell (e.g. unit quadrilateral)
           cubature points to a specific reference boundary cell
           (e.g. face of an hexahedron)
        */
        Intrepid::FieldContainer<value_t> update_boundary_cub( ushort_t cell_ord_ ){

            Intrepid::FieldContainer<value_t> bd_cub_pts_lifted_(bd_cub::numCubPoints, geo_map::spaceDim);
            Intrepid::FieldContainer<value_t> bd_cub_pts_(bd_cub::numCubPoints, geo_map::spaceDim-1);

            for (uint_t i=0; i<bd_cub::numCubPoints; ++i){
                for (uint_t j=0; j<geo_map::spaceDim-1; ++j){
                    bd_cub_pts_(i,j)=m_bd_cub_pts(i,j,0);
                }
            }

            //cubature points lifted to the reference cell
            //call the lift method on the parent-cell cubature class
            boundary_cubature<cubature<Order, parent_shape> >::lift( bd_cub_pts_lifted_, bd_cub_pts_, cell_ord_ );
            return bd_cub_pts_lifted_;
        }

        weights_storage_t & bd_cub_weights(){
                return m_bd_cub_weights;
        }

    };

    template<typename Rule>
    class boundary_discr{

    public:
        using value_t=float_type;
        using rule_t = Rule;
        using geo_map = typename rule_t::geo_map;
        static const enumtype::Shape parent_shape=rule_t::parent_shape;
        static const enumtype::Shape bd_shape=rule_t::bd_shape;
        // using bd_cub = typename rule_t::bd_cub;
        using cub = typename rule_t::bd_cub;
        typedef storage_t<layout_map<0,1,2> > grad_storage_t;
        typedef storage_t<layout_map<0,1,2> > phi_storage_t;
        typedef storage_t<layout_map<0,1,2> > tangent_storage_t;

        //private:

        rule_t & m_rule;
        grad_storage_t m_grad_at_cub_points;
        phi_storage_t m_phi_at_cub_points;
        tangent_storage_t m_ref_face_tg_u;
        tangent_storage_t m_ref_face_tg_v;
        ushort_t m_face_ord;

    public:

        boundary_discr(rule_t & rule_, ushort_t face_ord_):
            m_rule(rule_)
            , m_grad_at_cub_points(geo_map::basisCardinality, rule_t::bd_cub::numCubPoints, shape_property<rule_t::/*bd*/parent_shape>::dimension)
            , m_phi_at_cub_points(geo_map::basisCardinality, rule_t::bd_cub::numCubPoints, 1)
            , m_ref_face_tg_u(rule_t::bd_cub::numCubPoints, shape_property<rule_t::parent_shape>::dimension, 1)
            , m_ref_face_tg_v(rule_t::bd_cub::numCubPoints, shape_property<rule_t::parent_shape>::dimension, 1)
            , m_face_ord(face_ord_)
            {}

        void compute(Intrepid::EOperator const& operator_){
            auto face_quad_=m_rule.update_boundary_cub(m_face_ord);

            switch (operator_){
            case Intrepid::OPERATOR_GRAD :
            {
                Intrepid::FieldContainer<double> grad_at_cub_points(geo_map::basisCardinality, rule_t::bd_cub::numCubPoints, shape_property<rule_t::/*bd*/parent_shape>::dimension);
                // evaluate grad operator at the face cub points
                //NOTE: geo_map has the parent element basis, not the boundary one
                geo_map::hexBasis.getValues(grad_at_cub_points, face_quad_, Intrepid::OPERATOR_GRAD);

                for (uint_t l=0; l<geo_map::basisCardinality; ++l)
                    for (uint_t i=0; i<rule_t::bd_cub::numCubPoints; ++i)
                        for (uint_t j=0; j<shape_property<rule_t::/*bd*/parent_shape>::dimension; ++j)
                            m_grad_at_cub_points(l,i,j)=grad_at_cub_points(l,i,j);
                break;
            }
            case Intrepid::OPERATOR_VALUE :
            {
                Intrepid::FieldContainer<double> phi_at_cub_points(geo_map::basisCardinality, rule_t::bd_cub::numCubPoints);
                // evaluate grad operator at the face cub points
                geo_map::hexBasis.getValues(phi_at_cub_points, face_quad_, Intrepid::OPERATOR_VALUE);

                for (uint_t j=0; j<geo_map::basisCardinality; ++j)
                    for (uint_t i=0; i<rule_t::bd_cub::numCubPoints; ++i)
                        m_phi_at_cub_points(i,j,0)=phi_at_cub_points(i,j);
                break;
            }
            // case Intrepid::OPERATOR_VALUE :
            // {
            //     geo_map::hexBasis.getValues(m_phi_at_cub_points, face_quad_, Intrepid::OPERATOR_VALUE);
            //     break;
            // }
            default : assert(false);
            }
        }

        /**@brief get the 2 tangents on a point in the reference element*/
        void compute_tangents(){
            Intrepid::FieldContainer<double> tangent_u(rule_t::bd_cub::numCubPoints, shape_property<rule_t::parent_shape>::dimension);
            Intrepid::FieldContainer<double> tangent_v(rule_t::bd_cub::numCubPoints, shape_property<rule_t::parent_shape>::dimension);
            Intrepid::CellTools<value_t>::getReferenceFaceTangents(m_ref_face_tg_u, m_ref_face_tg_v, m_face_ord, geo_map::cell_t::value);

            for (uint_t i=0; i<rule_t::bd_cub::numCubPoints; ++i)
                    for (uint_t j=0; j<shape_property<rule_t::parent_shape>::dimension; ++j)
                    {
                        m_ref_face_tg_u(i,j,0)=tangent_u(i,j);
                        m_ref_face_tg_v(i,j,0)=tangent_v(i,j);
                    }
        }

        typename rule_t::weights_storage_t & bd_cub_weights() const
            {return m_rule.bd_cub_weights();}

        grad_storage_t & local_gradient()
            {return m_grad_at_cub_points;}

        phi_storage_t & phi()
            {return m_phi_at_cub_points;}


    };

}//namespace intrepid
}//namespace gridtools

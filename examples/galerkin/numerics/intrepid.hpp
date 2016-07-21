#pragma once

#ifdef __CUDACC__
#include <boost/shared_ptr.hpp>
#endif

// [includes]
#include <Intrepid_FunctionSpaceTools.hpp>
#include <Intrepid_Types.hpp>

#include <gridtools.hpp>
#include <stencil-composition/accessor.hpp>
#include <stencil-composition/interval.hpp>

#include <boost/type_traits.hpp>
#include "basis_functions.hpp"
#include "cubature.hpp"
// [includes]

//#define REORDER

namespace gdl{
namespace intrepid{
#ifdef __CUDACC__
    typedef gt::layout_map<2,1,0> layout3_t;
    typedef gt::layout_map<3,2,1,0> layout4_t;
#else
    typedef gt::layout_map<0,1,2> layout3_t;
    typedef gt::layout_map<0,1,2,3> layout4_t;
#endif
    /**
       @brief defining the finite element discretization

       Given the finite elements space definitions and the quadrature rule it instantiates the useful quantities
       (e.g. the values of the basis functions and of their derivatives) in the quadrature points.
     */
    template <typename FE, typename Cub>
    struct discretization{
        using fe=FE;
        using cub=Cub;
        static const enumtype::Shape parent_shape=FE::shape();
        static const enumtype::Shape bd_shape=shape_property<FE::shape()>::boundary;

        static const constexpr uint_t& basis_cardinality(){ return fe::basis_cardinality; }

        // [test]
        GRIDTOOLS_STATIC_ASSERT(fe::layout_t::template at_<0>::value < 3 && fe::layout_t::template at_<1>::value < 3 && fe::layout_t::template at_<2>::value < 3,
                                "the first three numbers in the layout_map must be a permutation of {0,1,2}. ");
        using weights_storage_t_info = storage_info_t< gt::pair<discretization<FE, Cub>, static_int<__COUNTER__> >, layout3_t >;
        using grad_storage_t_info = storage_info_t< gt::pair<discretization<FE, Cub>, static_int<__COUNTER__> >, layout3_t >;
        using basis_function_storage_t_info = storage_info_t<  gt::pair<discretization<FE, Cub>, static_int<__COUNTER__> >, layout3_t >;
        using cub_points_storage_t_info = storage_info_t< gt::pair<discretization<FE, Cub>, static_int<__COUNTER__> > , layout3_t >;

        using weights_storage_t        = storage_t< weights_storage_t_info > ;
        using grad_storage_t           = storage_t< grad_storage_t_info > ;
        using basis_function_storage_t = storage_t< basis_function_storage_t_info > ;
        using cub_points_storage_t     = storage_t< cub_points_storage_t_info > ;

    protected:
        std::vector<uint_t> m_permutations;
        cub_points_storage_t_info m_cub_points_s_info;
        weights_storage_t_info m_cub_weights_s_info;
        //these 2 are pointers, so that they get initialized on first touch
        std::unique_ptr<grad_storage_t_info> m_grad_at_cub_points_s_info;
        std::unique_ptr<basis_function_storage_t_info> m_phi_at_cub_points_s_info;

        cub_points_storage_t m_cub_points_s;
        weights_storage_t m_cub_weights_s;
        //these 2 are pointers, so that they get initialized on first touch
        std::unique_ptr<grad_storage_t> m_grad_at_cub_points_s;
        std::unique_ptr<basis_function_storage_t> m_phi_at_cub_points_s;

    public:

        cub_points_storage_t& cub_points()
        {return m_cub_points_s;}

        cub_points_storage_t const& get_cub_points() const
        {return m_cub_points_s;}

        weights_storage_t& cub_weights()
            {return m_cub_weights_s;}

        weights_storage_t const& get_cub_weights() const
            {return m_cub_weights_s;}

        grad_storage_t& grad()
            {
	         //If this assertion fails most probably you have not called
	         //compute with the OPERATOR_GRAD flag

                assert(m_grad_at_cub_points_s.get());
                return *m_grad_at_cub_points_s;
            }

        grad_storage_t const& grad() const
            {
	         //If this assertion fails most probably you have not called
	         //compute with the OPERATOR_GRAD flag

                assert(m_grad_at_cub_points_s.get());
                return *m_grad_at_cub_points_s;
            }


        basis_function_storage_t & val()
            {
	         //If this assertion fails most probably you have not called
	         //compute with the OPERATOR_VALUE flag
	         assert(m_phi_at_cub_points_s.get());
	         return *m_phi_at_cub_points_s;
            }


        basis_function_storage_t const& val() const
            {
	         //If this assertion fails most probably you have not called
	         //compute with the OPERATOR_VALUE flag
	         assert(m_phi_at_cub_points_s.get());
	         return *m_phi_at_cub_points_s;
            }


        discretization() :
            m_permutations( fe::basis_cardinality() )
            , m_cub_points_s_info(cub::numCubPoints(), fe::space_dim(),1)
            , m_cub_weights_s_info(cub::numCubPoints(),1,1)
            , m_grad_at_cub_points_s_info()//construct empty
            , m_phi_at_cub_points_s_info()//construct empty
            , m_cub_points_s(m_cub_points_s_info, 0., "cub points")
            , m_cub_weights_s(m_cub_weights_s_info, 0., "cub weights")
            , m_grad_at_cub_points_s()//construct empty
            , m_phi_at_cub_points_s()//construct empty
            {
                for(uint_t i=0; i< fe::basis_cardinality(); ++i){
                    m_permutations[i]=i;
                }

            }

        void compute(Intrepid::EOperator const& operator_){

            // storage_t<layout_map<0,1,2> > cub_points_i(m_cub_points_s, 2);
            Intrepid::FieldContainer<double> cub_points_i(cub::numCubPoints(), fe::space_dim());

            // storage_t<layout_map<0,1,2> > cub_weights_i(m_cub_weights_s, 1);
            Intrepid::FieldContainer<double> cub_weights_i(cub::numCubPoints());

            // storage_t<layout_map<0,1,2> > grad_at_cub_points_i(m_grad_at_cub_points_s);
            Intrepid::FieldContainer<double> grad_at_cub_points_i(fe::basis_cardinality(), cub::numCubPoints(), fe::space_dim());

            // retrieve cub points and weights
            cub::cub()->getCubature(cub_points_i, cub_weights_i);

            //copy the values
            for (uint_t q=0; q<cub::numCubPoints(); ++q)
            {
                m_cub_weights_s(q,0,0)=cub_weights_i(q);
                for (uint_t j=0; j<fe::space_dim(); ++j)
                    m_cub_points_s(q,j,0)=cub_points_i(q,j);
            }

            switch (operator_){
            case Intrepid::OPERATOR_GRAD :
            {
                m_grad_at_cub_points_s_info=std::unique_ptr
                    <grad_storage_t_info>
                    (new grad_storage_t_info
                     (fe::basis_cardinality(), cub::numCubPoints(), fe::space_dim()));

                m_grad_at_cub_points_s=std::unique_ptr
                    <grad_storage_t>
                    (new grad_storage_t
                     (*m_grad_at_cub_points_s_info, 0., "grad at cub points"));

                // evaluate grad operator at cub points
                fe::hex_basis().getValues(grad_at_cub_points_i, cub_points_i, Intrepid::OPERATOR_GRAD);

                for (uint_t i=0; i<fe::basis_cardinality(); ++i)
                	for (uint_t q=0; q<cub::numCubPoints(); ++q)
                            for (uint_t j=0; j<fe::space_dim(); ++j)
                                (*m_grad_at_cub_points_s)(m_permutations[i],q,j)=grad_at_cub_points_i(i,q,j);

                // for (uint_t q=0; q<cub::numCubPoints(); ++q)
                //     for (uint_t j=0; j<fe::spaceDim; ++j)
                //     {
                //         (*m_grad_at_cub_points_s)(2,q,j)=grad_at_cub_points_i(3,q,j);
                //         (*m_grad_at_cub_points_s)(3,q,j)=grad_at_cub_points_i(2,q,j);
                //         (*m_grad_at_cub_points_s)(6,q,j)=grad_at_cub_points_i(7,q,j);
                //         (*m_grad_at_cub_points_s)(7,q,j)=grad_at_cub_points_i(6,q,j);
                //     }

                break;
            }

            case Intrepid::OPERATOR_VALUE :
            {
                m_phi_at_cub_points_s_info=std::unique_ptr
                    <basis_function_storage_t_info>
                    (new basis_function_storage_t_info
                     (fe::basis_cardinality(), cub::numCubPoints(), 1));

                m_phi_at_cub_points_s=std::unique_ptr
                    <basis_function_storage_t>
                    (new basis_function_storage_t
                     (*m_phi_at_cub_points_s_info, 0., "phi at cub points"));

                Intrepid::FieldContainer<double> phi_at_cub_points_i(fe::basis_cardinality()
                                                                     , cub::numCubPoints());

                fe::hex_basis().getValues(phi_at_cub_points_i, cub_points_i, Intrepid::OPERATOR_VALUE);

                //copy out the values
                for (uint_t q=0; q<cub::numCubPoints(); ++q)
                    // for (uint_t j=0; j<fe::spaceDim; ++j)
                    for (uint_t i=0; i<fe::basis_cardinality(); ++i){
                        (*m_phi_at_cub_points_s)(m_permutations[i],q,0)=phi_at_cub_points_i(i,q);
                    }
                break;
            }
            default : assert(false);
            }

        }

        std::vector<uint_t> const& get_ordering(){return m_permutations;}

    };


    /**
       @brief discretization of the local-to-global map

       This map is what connects the reference to the actual configuration. All the quantities and integrals computed
       in the reference element get eventually interpolated in the actual configuration using this map. The case in which this class
       uses the same basis functions as the finite element discretization is called "isoparametric".
     */
    template < typename GeoMap, typename Cubature >
    struct unstructured_geometry : public discretization<GeoMap, Cubature>{

        using geo_map=GeoMap;
        using super=discretization<GeoMap, Cubature>;

        /**
           @brief constructor

           NOTE: the computation of the derivatives is automatically triggered,
           since it is always used to compute the metric tensor (i.e. the first fundamental form)
         */
        unstructured_geometry()
            {
                super::compute(Intrepid::OPERATOR_GRAD);
            }

    };

    // TODO: not sure if we want a "is a" relation with the geometry struct
    // TODO: can we remove the REORDER ifdef in the structured geometry derived class?
    /**
     * @brief discretization of the local-to-global map, structured mesh case (allowing dof ordering)
     */
    template < typename GeoMap, typename Cubature >
    struct geometry : public discretization<GeoMap, Cubature>{

        using super=discretization<GeoMap, Cubature>;

        using geo_map = GeoMap;
        using bd_geo_map = reference_element<geo_map::order()
                                             , geo_map::basis()
                                             , shape_property<geo_map::shape()>::boundary>;

        using local_grid_t_info = storage_info< __COUNTER__, layout3_t >;
        using local_grid_t = storage_t< local_grid_t_info >;
        local_grid_t_info  m_local_grid_s_info;
        local_grid_t  m_local_grid_s;

#ifdef REORDER
        using local_grid_reordered_t = storage_t< local_grid_t_info >;
        local_grid_reordered_t m_local_grid_reordered_s;
#endif

        /**
           @brief constructor

           NOTE: the computation of the derivatives is automatically triggered,
           since it is always used to compute the metric tensor (i.e. the first fundamental form)
         */
        geometry() :
            //create the local grid
            m_local_grid_s_info(geo_map::basis_cardinality(), geo_map::space_dim(),1)
            , m_local_grid_s(m_local_grid_s_info, 0., "local grid")
#ifdef REORDER
            , m_local_grid_reordered_s(m_local_grid_s_info, 0., "local grid reordered")
#endif
            {
                // storage_t<layout_map<0,1,2> > local_grid_i(m_local_grid_s, 2);
                Intrepid::FieldContainer<double> local_grid_i(geo_map::basis_cardinality(), geo_map::space_dim());
                geo_map::hex_basis().getDofCoords(local_grid_i);
                for (uint_t i=0; i<geo_map::basis_cardinality(); ++i)
                    for (uint_t j=0; j<geo_map::space_dim(); ++j)
                        m_local_grid_s(i,j,0)=local_grid_i(i,j);
#ifdef REORDER

                //! [reorder]
                std::vector<gt::float_type> to_reorder( geo_map::basis_cardinality() );
                //sorting the a vector containing the point coordinates with priority i->j->k, and saving the permutation

                // fill in the reorder vector such that the larger numbers correspond to larger strides
                for(uint_t i=0; i<geo_map::basis_cardinality(); ++i){
		    to_reorder[i]=(m_local_grid_s(i,geo_map::layout_t::template at_<0>::value*(geo_map::space_dim()-2),0)+2)*16 +
                        (m_local_grid_s(i,geo_map::layout_t::template at_<1>::value,0)+2)*4 +
                        (m_local_grid_s(i,geo_map::layout_t::template at_<2>::value,0)+2);
                    // m_permutations[i]=i;
                }

                std::sort(this->m_permutations.begin(), this->m_permutations.end(),
                  [&to_reorder](uint_t a, uint_t b){
                      return to_reorder[a]<to_reorder[b];
                  } );

                // storage_t<layout_map<0,1,2> >::storage_t  local_grid_reordered_s(geo_map::basis_cardinality(), geo_map::space_dim(),1);
                // storage_t<layout_map<0,1,2> >  local_grid_reordered_i(m_local_grid_reordered_s, 2);
                uint_t D=geo_map::basis_cardinality();

                //applying the permutation to the grid
                for(uint_t i=0; i<D; ++i){//few redundant loops
                	for(uint_t j=0; j<geo_map::space_dim(); ++j)
                        {
                            m_local_grid_reordered_s(i, j, 0)=m_local_grid_s(this->m_permutations[i],j,0);
                        }
                }
                //! [reorder]
#endif
            }

#ifdef REORDER
        local_grid_reordered_t const& reordered_grid() const { return m_local_grid_reordered_s; }
#else
        local_grid_t const& grid() const {return m_local_grid_s;}
#endif

    };



    template<typename GeoMap, ushort_t Order>
    class boundary_cub // : public Base
    {
    public:

        using weights_storage_t_info = storage_info< __COUNTER__,layout3_t >;
        using cub_points_storage_t_info = storage_info< __COUNTER__, layout3_t >;
        using weights_storage_t        = storage_t< weights_storage_t_info > ;
        using cub_points_storage_t     = storage_t< cub_points_storage_t_info > ;

        // typedef storage_t<layout_map<0,1,2> > points_lifted_storage_t;

        using geo_map = GeoMap;
        using value_t=float_type;
        static const enumtype::Shape parent_shape=geo_map::shape();
        static const enumtype::Shape bd_shape=shape_property<geo_map::shape()>::boundary;
        static const ushort_t n_sub_cells=shape_property<geo_map::shape()>::n_sub_cells;
        using bd_cub = cubature<Order, bd_shape>;

        //private:

        cub_points_storage_t_info m_bd_cub_pts_info;
        weights_storage_t_info m_bd_cub_weights_info;
        cub_points_storage_t m_bd_cub_pts;
        weights_storage_t m_bd_cub_weights;
        // points_lifted_storage_t  m_bd_cub_pts_lifted;

    public:
        boundary_cub()
            :
            m_bd_cub_pts_info(bd_cub::numCubPoints(), geo_map::space_dim()-1, 1)
            , m_bd_cub_weights_info(bd_cub::numCubPoints(), 1, 1)
            , m_bd_cub_pts(m_bd_cub_pts_info, 0., "bd cub points")
            , m_bd_cub_weights(m_bd_cub_weights_info, 0., "bd cub weights")
            // , m_bd_cub_pts_lifted(bd_cub::numCubPoints, geo_map::space_dim(), 1)
            {
                Intrepid::FieldContainer<value_t> bd_cub_pts_(bd_cub::numCubPoints(), geo_map::space_dim()-1);
                Intrepid::FieldContainer<value_t> bd_cub_weights_(bd_cub::numCubPoints());

                bd_cub::cub()->getCubature(bd_cub_pts_, bd_cub_weights_);

                for (uint_t i=0; i<bd_cub::numCubPoints(); ++i){
                    m_bd_cub_weights(i,0,0)=bd_cub_weights_(i);
                    for (uint_t j=0; j<geo_map::space_dim()-1; ++j){
                        m_bd_cub_pts(i,j,0)=bd_cub_pts_(i,j);
                    }
                }
            }

        /**@brief performs the lift of the reference subcell (e.g. unit quadrilateral)
           cubature points to a specific reference boundary cell
           (e.g. face of an hexahedron)
        */
        Intrepid::FieldContainer<value_t> update_boundary_cub( ushort_t cell_ord_ ){

            Intrepid::FieldContainer<value_t> bd_cub_pts_lifted_(bd_cub::numCubPoints(), geo_map::space_dim());
            Intrepid::FieldContainer<value_t> bd_cub_pts_(bd_cub::numCubPoints(), geo_map::space_dim()-1);

            for (uint_t i=0; i<bd_cub::numCubPoints(); ++i){
                for (uint_t j=0; j<geo_map::space_dim()-1; ++j){
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

    /**
       @brief discretization of the boundary entities

       given the quadrature rule, it defines the discretization of the boundary entities
     */
    template<typename Rule, ushort_t Boundaries = shape_property<Rule::parent_shape>::n_sub_cells>
    class boundary_discr{

    public:
        static const constexpr ushort_t s_num_boundaries = Boundaries;
        using value_t=float_type;
        using rule_t = Rule;
        using geo_map = typename rule_t::geo_map;
        static const enumtype::Shape parent_shape=rule_t::parent_shape;
        static const enumtype::Shape bd_shape=rule_t::bd_shape;
        // using bd_cub = typename rule_t::bd_cub;
        using cub = typename rule_t::bd_cub;
        using bd_geo_map=reference_element<geo_map::order(), geo_map::basis(), bd_shape>;

        using grad_storage_t_info = storage_info< __COUNTER__,layout4_t >;
        using basis_function_storage_t_info = storage_info< __COUNTER__,layout3_t >;
        using tangent_storage_t_info = storage_info< __COUNTER__,layout3_t>;

        using grad_storage_t           = storage_t< grad_storage_t_info > ;
        using basis_function_storage_t = storage_t< basis_function_storage_t_info > ;
        using tangent_storage_t = storage_t<tangent_storage_t_info >;
        using weights_storage_t = typename rule_t::weights_storage_t;
        //private:
        static const int n_sub_cells = shape_property<parent_shape>::n_sub_cells;

        rule_t & m_rule;


        grad_storage_t_info m_grad_at_cub_points_info;
        basis_function_storage_t_info m_phi_at_cub_points_info;
        tangent_storage_t_info m_ref_face_tg_info;

        grad_storage_t m_grad_at_cub_points;
        basis_function_storage_t m_phi_at_cub_points;
        tangent_storage_t m_ref_face_tg_u;
        tangent_storage_t m_ref_face_tg_v;
        tangent_storage_t m_ref_normals;
        gt::array<ushort_t, Boundaries> m_face_ord;
        bool m_tangent_computed;

    public:

        template<typename ... UShort>
        boundary_discr(rule_t & rule_, UShort ... face_ord_):
            m_rule(rule_)
            , m_grad_at_cub_points_info(geo_map::basis_cardinality(), rule_t::bd_cub::numCubPoints(), shape_property<rule_t::/*bd*/parent_shape>::dimension, sizeof ... (UShort))
            , m_phi_at_cub_points_info(geo_map::basis_cardinality(), rule_t::bd_cub::numCubPoints(), sizeof ... (UShort))
            , m_ref_face_tg_info(shape_property<rule_t::parent_shape>::dimension, sizeof ... (UShort), 1u)
            , m_grad_at_cub_points(m_grad_at_cub_points_info, 0., "bd grad at cub")
            , m_phi_at_cub_points(m_phi_at_cub_points_info, 0., "bd phi at cub")
            , m_ref_face_tg_u(m_ref_face_tg_info, 0., "tg u")
            , m_ref_face_tg_v(m_ref_face_tg_info, 0., "tg v")
            , m_ref_normals(m_ref_face_tg_info, 0., "reference normals")
            , m_face_ord{(ushort_t) face_ord_ ...}
            , m_tangent_computed(false)
            {
                GRIDTOOLS_STATIC_ASSERT(gt::accumulate(gt::logical_and(), boost::is_integral<UShort>::type::value ...),
                                        "the face ordinals must be of integral type in the boundary discretization constructor");

            }

        void compute(Intrepid::EOperator const& operator_, std::vector<uint_t> ordering = {}){


            compute_tangents();
            compute_normals();


            if(!ordering.size())
            {
                ordering.resize(geo_map::basis_cardinality());
                for (uint_t dof_=0; dof_<geo_map::basis_cardinality(); ++dof_){
                    ordering[dof_]=dof_;
                }
            }

            for(ushort_t face_=0; face_< m_face_ord.size(); ++face_){
                auto face_quad_=m_rule.update_boundary_cub(m_face_ord[face_]);

                // std::cout<<"face ord: "<<face_<<"\n";
                // for(int i=0; i<4; ++i)
                //     for(int j=0; j<3; ++j)
                //         std::cout<<" i "<< i << " j "<< j  <<" "<<face_quad_(i,j)<<"\n";
                switch (operator_){
                case Intrepid::OPERATOR_GRAD :
                {
                    Intrepid::FieldContainer<double> grad_at_cub_points(geo_map::basis_cardinality(), rule_t::bd_cub::numCubPoints(), shape_property<rule_t::/*bd*/parent_shape>::dimension);
                    // evaluate grad operator at the face cub points
                    //NOTE: geo_map has the parent element basis, not the boundary one
                    geo_map::hex_basis().getValues(grad_at_cub_points, face_quad_, Intrepid::OPERATOR_GRAD);

                    for (uint_t l=0; l<geo_map::basis_cardinality(); ++l)
                        for (uint_t i=0; i<rule_t::bd_cub::numCubPoints(); ++i)
                            for (uint_t j=0; j<shape_property<rule_t::/*bd*/parent_shape>::dimension; ++j)
                                m_grad_at_cub_points(
                                    ordering[l]
                                    ,i,j,face_)=grad_at_cub_points(l,i,j);
                    break;
                }
                case Intrepid::OPERATOR_VALUE :
                {
                    Intrepid::FieldContainer<double> phi_at_cub_points(geo_map::basis_cardinality(), rule_t::bd_cub::numCubPoints());
                    // evaluate grad operator at the face cub points
                    geo_map::hex_basis().getValues(phi_at_cub_points, face_quad_, Intrepid::OPERATOR_VALUE);

                    for (uint_t i=0; i<geo_map::basis_cardinality(); ++i)
                        for (uint_t j=0; j<rule_t::bd_cub::numCubPoints(); ++j)
                        {
                            m_phi_at_cub_points(
                                ordering[i]
                                ,j,face_)=phi_at_cub_points(i,j);
                        }
                    break;
                }

                default :
                {
                    std::cout<<"Operator not supported"<<std::endl;
                    assert(false);
                }
                }
            }
        }

        /**@brief get the 2 tangents on the faces of the reference element

           TODO underlying hypothesis that parent shape is 3D
         */
        void compute_tangents(){

            for(ushort_t face_=0; face_< m_face_ord.size(); ++face_){
                Intrepid::FieldContainer<double> tangent_u(shape_property<rule_t::parent_shape>::dimension);
                Intrepid::FieldContainer<double> tangent_v(shape_property<rule_t::parent_shape>::dimension);
                Intrepid::CellTools<value_t>::getReferenceFaceTangents(tangent_u, tangent_v, m_face_ord[face_], geo_map::cell_t::value);

                for (uint_t j=0; j<shape_property<rule_t::parent_shape>::dimension; ++j)
                {
                    m_ref_face_tg_u(j,face_,0)=tangent_u(j);
                    m_ref_face_tg_v(j,face_,0)=tangent_v(j);
                }
            }
            m_tangent_computed=true;
        }


        /**@brief get the normals to the faces of the reference element

           TODO underlying hypothesis that parent shape is 3D
         */
        void compute_normals(){
            assert(m_tangent_computed);

            for(ushort_t face_=0; face_< m_face_ord.size(); ++face_){
                gt::array<double, 3> tg_u{m_ref_face_tg_u(0,face_,0), m_ref_face_tg_u(1,face_,0), m_ref_face_tg_u(2,face_,0)};
                gt::array<double, 3> tg_v{m_ref_face_tg_v(0,face_,0), m_ref_face_tg_v(1,face_,0), m_ref_face_tg_v(2,face_,0)};
                gt::array<double, 3> normal(vec_product(tg_u, tg_v));

                for (uint_t j=0; j<shape_property<rule_t::parent_shape>::dimension; ++j)
                {
                    m_ref_normals(j,face_,0)=normal[j];
                }
            }
        }

        typename rule_t::weights_storage_t & bd_cub_weights()
        {return m_rule.bd_cub_weights();}

        grad_storage_t & grad()
        {return m_grad_at_cub_points;}

        basis_function_storage_t & val()
        {return m_phi_at_cub_points;}

        /**
           @brief returns the normals to the element boundaries in the reference configuration
         */
        tangent_storage_t & ref_normals()
        {return m_ref_normals;}

        /**
           @brief returns the number of boundaries for which the discretization is defined
           (normals, tangents, integration rules, ...)
         */
        ushort_t n_boundaries()
        {return m_face_ord.size();}


    };

    //external visibility
    template<typename Rule, ushort_t Boundaries>
    const constexpr ushort_t boundary_discr< Rule,  Boundaries>::s_num_boundaries;

}//namespace intrepid
}//namespace gdl

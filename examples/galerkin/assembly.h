#pragma once

// [includes]
#include <stencil-composition/make_computation.hpp>
#include "basis_functions.h"
// [includes]
#ifdef CXX11_ENABLED

using namespace gridtools;
using namespace enumtype;
using namespace expressions;

struct assembly{

    //                      dims  x y z  qp
    //                   strides  1 x xy xyz
    typedef gridtools::layout_map<0,1,2,3> layout_t;
    typedef gridtools::BACKEND::storage_type<float_type, layout_t >::type storage_type;
    typedef gridtools::layout_map<0,1,2,3,4,5> layout_jacobian_t;
    typedef gridtools::BACKEND::storage_type<float_type, layout_jacobian_t >::type jacobian_type;
    typedef gridtools::layout_map<0,1,2,3,4> layout_grid_t;
    typedef gridtools::BACKEND::storage_type<float_type, layout_grid_t >::type grid_type;
    static const int_t edge_points=fe::hypercube_t::boundary_w_dim<1>::n_points::value;


private:

    uint_t m_d1, m_d2, m_d3;
    grid_type m_grid;
    jacobian_type m_jac;
    storage_type m_jac_det;
    storage_type m_f;
    jacobian_type m_result;

public:

    assembly(uint_t d1, uint_t d2, uint_t d3 ): m_d1(d1)
                                              , m_d2(d2)
                                              , m_d3(d3)
                                              , m_grid(d1, d2, d3, geo_map::basisCardinality, 3)
                                              , m_jac(d1, d2, d3, cubature::numCubPoints, 3, 3)
                                              , m_jac_det(d1, d2, d3, cubature::numCubPoints)
                                              , m_f(d1, d2, d3, cubature::numCubPoints)
                                              , m_result(d1, d2, d3, edge_points, edge_points, edge_points)
        {}

    jacobian_type const& get_jac() const {return m_jac;}
    storage_type const& get_jac_det() const {return m_jac_det;}
    grid_type const& get_grid() const {return m_grid;}


    // /** updates the values of the Jacobian matrix. The Jacobian matrix, component (i,j) in the quadrature point q, is computed given the geometric map discretization as \f$ J(i,j,q)=\sum_k\frac{\partial \phi_i(x_k,q)}{\partial x_j} x_k \f$
    //     where x_k are the points in the geometric element*/
    // struct initialize{
    //     typedef accessor<0, range<0,0,0,0> , 5> const grid_points;
    //     typedef accessor<1, range<0,0,0,0> , 3> element_grid;
    //     typedef accessor<2, range<0,0,0,0> , 4> position;

    //     //the iterate_domain knows which are the strides/dims: implement a way to extract them
    //     //uniform 3D grid
    //     // m_grid.initialize([d1, d2, d3, &element_grid](int i, int j, int k, int point, int l){ return (l==0 ? (float)(i/d1) : l==2 ? (float)(j/d2) : (float)(k/d3)) + element_grid(point, 0)/d1 + element_grid(point, 1)/d2 + element_grid(point, 2)/d3; });

    //     template <typename Evaluation>
    //     GT_FUNCTION
    //     static void Do(Evaluation const & eval, x_interval) {

    //         for (int_t iterNode=0; iterNode < geo_map::basisCardinality ; ++iterNode)
    //         {
    //             eval( grid_points(0,0,0,iterNode,0) ) +=  position(0,0,0,0) + element_grid(point, 0) ;
    //             eval( grid_points(0,0,0,iterNode,1) ) +=  position(0,0,0,1) + element_grid(point, 1) ;
    //             eval( grid_points(0,0,0,iterNode,2) ) +=  position(0,0,0,2) + element_grid(point, 2) ;
    //         }
    //     }
    // };


    /** updates the values of the Jacobian matrix. The Jacobian matrix, component (i,j) in the quadrature point q, is computed given the geometric map discretization as \f$ J(i,j,q)=\sum_k\frac{\partial \phi_i(x_k,q)}{\partial x_j} x_k \f$
        where x_k are the points in the geometric element*/
    struct update_jac{
        typedef accessor<0, range<0,0,0,0> , 5> const grid_points;
        typedef accessor<1, range<0,0,0,0> , 6> jac;
        typedef accessor<2, range<0,0,0,0> , 3> const dphi;
        typedef boost::mpl::vector< grid_points, jac, dphi> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            Dimension<4>::Index qp;
            Dimension<5>::Index dimx;
            Dimension<6>::Index dimy;

            for (int_t iterNode=0; iterNode < geo_map::basisCardinality ; ++iterNode)
            {//reduction/gather
                for(short_t icoor=0; icoor< 3; ++icoor)
                {
                    for(short_t jcoor=0; jcoor< 3; ++jcoor)
                    {
                        // eval( jac(dimx+icoor, dimy+jcoor, qp+iter_quad) )=0.;
                        for(short_t iter_quad=0; iter_quad< cubature::numCubPoints/*quad_pts*/; ++iter_quad)
                        {
                            eval( jac(dimx+icoor, dimy+jcoor, qp+iter_quad) ) += eval(grid_points(0,0,0, iterNode, icoor) * !dphi(iterNode, iter_quad, jcoor) );
                        }
                    }
                }
            }


        }
    };



    /** updates the values of the Jacobian matrix. The Jacobian matrix, component (i,j) in the quadrature point q, is computed given the geometric map discretization as \f$ J(i,j,q)=\sum_k\frac{\partial \phi_i(x_k,q)}{\partial x_j} x_k \f$
        where x_k are the points in the geometric element*/
    struct det{
        typedef accessor<0, range<0,0,0,0> , 6> const jac;
        typedef accessor<1, range<0,0,0,0> , 4> const jac_det;
        typedef boost::mpl::vector< jac, jac_det > arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            Dimension<4>::Index qp;
            Dimension<5>::Index dimx;
            Dimension<6>::Index dimy;

            for(short_t q=0; q< cubature::numCubPoints; ++q)
            {
                eval( jac_det(qp+q) )= eval(
                    jac(        qp+q)*jac(dimx+1, dimy+1, qp+q)*jac(dimx+2, dimy+2, qp+q) +
                    jac(dimx+1, qp+q)*jac(dimx+2, dimy+1, qp+q)*jac(dimy+2,         qp+q) +
                    jac(dimy+1, qp+q)*jac(dimx+1, dimy+2, qp+q)*jac(dimx+2,         qp+q) -
                    jac(dimy+1, qp+q)*jac(dimx+1,         qp+q)*jac(dimx+2, dimy+2, qp+q) -
                    jac(        qp+q)*jac(dimx+2, dimy+1, qp+q)*jac(dimx+1, dimy+2, qp+q) -
                    jac(dimy+2, qp+q)*jac(dimx+1, dimy+1, qp+q)*jac(dimx+2,         qp+q)
                    );
            }
        }
    };

    struct integration {
        typedef accessor<0, range<0,0,0,0> , 3> const dphi;
        typedef accessor<1, range<0,0,0,0> , 4> const jac_det;
        typedef accessor<2, range<0,0,0,0> , 4> const f;
        typedef accessor<3, range<0,0,0,0> , 6> result;
        typedef boost::mpl::vector<dphi, jac_det, f, result> arg_list;
        using quad=Dimension<4>;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            quad::Index qp;
            static int_t dd=fe::hypercube_t::boundary_w_codim<2>::n_points::value;
            //projection of f on a (e.g.) P1 FE space ReferenceFESpace1:
            //loop on quadrature nodes, and on nodes of the P1 element (i,j,k) with i,j,k\in {0,1}
            //computational complexity in the order of  {(nDOF) x (nDOF) x (nq)}
            for(short_t P_i=0; P_i<dd; ++P_i)
                for(short_t P_j=0; P_j<dd; ++P_j)
                    for(short_t P_k=0; P_k<dd; ++P_k)
                        for(short_t J=0; J<fe::basisCardinality; ++J)
                            for(short_t q=0; q<cubature::numCubPoints; ++q){
                                uint_t dof=P_i*dd+P_j*1+P_k;
                                eval(result(0,0,0,P_i, P_j, P_k)) +=
                                    eval(!dphi(dof,q,0)*!dphi(J,q,0)*jac_det(qp+q)*f(qp+q))/8;
                            }
        }
    };


    template <typename ReferenceFESpace1, typename ReferenceFESpace2>
    struct assembly {
        typedef accessor<0, range<0,0,0,0> , 4> const in;
        typedef accessor<1, range<0,0,0,0> , 4> const out;
        //local_grid must be +-1 on the boundary and 0 on the internal
        // typedef accessor<2, range<0,0,0,0> , 2> const local_grid;
        typedef boost::mpl::vector<in, out> arg_list;
        using quad=Dimension<4>;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            x::Index i;
            y::Index j;
            z::Index k;
            // Dimension<4>::Index point;
            Dimension<4>::Index loc_x;
            Dimension<5>::Index loc_y;
            Dimension<6>::Index loc_z;

            // assembly : this part is specific for tensor product topologies
            static int_t bd_dim=fe::hypercube_t::boundary_w_codim<2>::n_points::value;

            //boundary dofs
            for(short_t I=0; I<bd_dim; I++)
                for(short_t J=0; J<bd_dim; J++)
                {
                    eval(out(loc_y+I, loc_z+J)) += eval(out(i-1, loc_x+bd_dim, loc_y+I, loc_z+J));
                    eval(out(loc_x+I, loc_z+J)) += eval(out(j-1, loc_y+bd_dim, loc_x+I, loc_z+J));
                    eval(out(loc_x+I, loc_y+J)) += eval(out(k-1, loc_z+bd_dim, loc_x+I, loc_y+J));
                }

            // // internal dofs
            // for(short_t I=1; I<bd_dim-1; I++)
            // for(short_t J=1; J<bd_dim-1; J++)
            // for(short_t K=1; K<bd_dim-1; K++)
            //     eval(out(loc_x+I, loc_y+J, loc_z+K)) = eval(in(loc_x+I, loc_y+J, loc_z+K));

            // for(short_t I=0; I<geo_map::basisCardinality; ++I)
            // {
            //     eval(out(point+counter)) = eval(out());
            //     //check if the dof is on the lower or left boundary (I could know this a-priori)
            //     int_t offset_i=eval(local_grid(I,0));
            //     int_t offset_j=eval(local_grid(I,1));
            //     int_t offset_k=eval(local_grid(I,2));

            //     eval(out(point+counter)) += eval(out(i+offset_i, point+I));
            //     eval(out(point+counter)) += eval(out(j+offset_j, point+I));
            //     eval(out(point+counter)) += eval(out(k+offset_k, point+I));
            //     counter++;
            // }
        }
    };

    // template<typename T, typename U>
    // std::ostream& operator<<(std::ostream& s, integration<T,U> const) {
    //     return s << "integration";
    // }

    template <typename GridType, typename StorageGradType>
    bool compute( GridType& element_grid, StorageGradType& local_gradient ){

        uint_t d1=m_d1;
        uint_t d2=m_d2;
        uint_t d3=m_d3;

        //constructing the grid
        for (uint_t i=0; i<d1; i++)
            for (uint_t j=0; j<d2; j++)
                for (uint_t k=0; k<d3; k++)
                    for (uint_t point=0; point<fe::basisCardinality; point++)
                    {
                        m_grid( i,  j,  k,  point,  0)= i + element_grid(point, 0);
                        m_grid( i,  j,  k,  point,  1)= j + element_grid(point, 1);
                        m_grid( i,  j,  k,  point,  2)= k + element_grid(point, 2);
                    }

        m_result.initialize(0.);
        m_f.initialize(0.);

        //I might want to store the jacobian as a temporary storage (will use less memory but
        //not reusing when I need the Jacobian for multiple things).

        /**I have to define here the placeholders to the storages used: the temporary storages get internally managed, while
           non-temporary ones must be instantiated by the user below. In this example all the storages are non-temporaries.*/
        // typedef arg<0, basis_func_type > p_phi;
        // typedef arg<1, basis_func_type > p_psi;
        typedef arg<0, grid_type >       p_grid_points;
        typedef arg<1, jacobian_type >   p_jac;
        typedef arg<2, storage_type >   p_jac_det;
        typedef arg<3, StorageGradType > p_dphi;
        typedef arg<4, storage_type >    p_f;
        typedef arg<5, jacobian_type >    p_result;

        typedef boost::mpl::vector<p_grid_points, p_jac, p_jac_det, p_dphi, p_f, p_result> accessor_list;


        gridtools::domain_type<accessor_list> domain(boost::fusion::make_vector(&m_grid, &m_jac, &m_jac_det, &local_gradient, &m_f, &m_result));

        /**
           - Definition of the physical dimensions of the problem.
           The coordinates constructor takes the horizontal plane dimensions,
           hile the vertical ones are set according the the axis property soon after
        */
        uint_t di[5] = {0, 0, 0, m_d1-1, m_d1};
        uint_t dj[5] = {0, 0, 0, m_d2-1, m_d2};
        gridtools::coordinates<axis> coords(di,dj);
        coords.value_list[0] = 0;
        coords.value_list[1] = m_d3-1;

#ifdef __CUDACC__
        computation* fe_comp =
#else
            boost::shared_ptr<gridtools::computation> fe_comp =
#endif
            make_computation<gridtools::BACKEND, layout_t>
            (
                make_mss
                (
                    execute<forward>(),
                    make_esf<update_jac>( p_grid_points(), p_jac(), p_dphi())
                    , make_esf<det>(p_jac(), p_jac_det())
                    , make_esf<integration>(p_dphi(), p_jac_det(), p_f(), p_result())
                    ),
                domain, coords);

        fe_comp->ready();
        fe_comp->steady();
        fe_comp->run();
        fe_comp->finalize();

        // result.print();

        return true;
    }

}; //namespace assembly
#endif //CXX11_ENABLED

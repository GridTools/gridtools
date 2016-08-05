/**
\file
*/
#define PEDANTIC_DISABLED
#include "../numerics/assembly.hpp"
#include "../functors/matvec.hpp"

namespace functors{
    namespace mass_IGA{
// [integration]
/** The following functor performs the assembly of an elemental laplacian.
 */
        template <typename FE, typename Cubature>
        struct mass {
            using fe=FE;
            using cub=Cubature;

            //![accessors]
            using jac_det =accessor<0, enumtype::in, extent<0,0,0,0> , 4> const;
            using weights =accessor<1, enumtype::in, extent<0,0,0,0> , 3> const;
            using mass_t   =accessor<2, enumtype::inout, extent<0,0,0,0> , 5> ;
            using phi    =accessor<3, enumtype::in, extent<0,0,0,0> , 3> const;
            using psi    =accessor<4, enumtype::in, extent<0,0,0,0> , 3> const;
            using arg_list= boost::mpl::vector<jac_det, weights, mass_t, phi,psi> ;
            //![accessors]

            //![Do_stiffness]
            template <typename Evaluation>
            GT_FUNCTION
            static void Do(Evaluation const & eval, x_interval) {
                //quadrature points dimension
                dimension<4>::Index qp;
                uint_t const num_cub_points=eval.template get_storage_dim<3>(jac_det());
                uint_t const basis_cardinality=eval.template get_storage_dim<0>(psi());


#ifndef __CUDACC__
                assert(num_cub_points==cub::numCubPoints());
                assert(basis_cardinality==fe::basisCardinality);
#endif

                //loop on quadrature nodes, and on nodes of the P1 element (i,j,k) with i,j,k\in {0,1}
                for(int P_i=0; P_i<basis_cardinality; ++P_i) // current dof
                {
                    for(int Q_i=0; Q_i<basis_cardinality; ++Q_i)
                    {//other dofs whose basis function has nonzero support on the element
                        for(int q=0; q<num_cub_points; ++q){
                            eval(mass_t(0,0,0,P_i,Q_i))  +=
                                eval(!phi(P_i,q,0)*(!psi(Q_i,q,0))*jac_det(qp+q)*!weights(q,0,0));
                        }
                    }
                }
            }
            //![Do_stiffness]
        };
//[integration]
    }//namespace functors

    int main(){

        //Basis functions:
        // order:
        static const int P=2;


        //b-splines in 1 D
        //the finite elements definitions
        using fe=reference_element<P, enumtype::BSplines, Line>;
        //cubature rule
        using cub=cubature<P, fe::shape>;
        //finite element discretization
        using fe_t = intrepid::discretization<fe, cub>;
        //![instantiation]
        //finite element
        fe_t fe_;

        //b-splines in 2 D
        //the finite elements definitions
        using fe2=reference_element<P, enumtype::BSplines, Quad>;
        //cubature rule
        using cub2=cubature<P, fe2::shape>;
        //finite element discretization
        using fe2_t = intrepid::discretization<fe2, cub2>;
        //![instantiation]
        //finite element
        fe2_t fe2_;

        //b-splines in 3 D
        //the finite elements definitions
        using fe3=reference_element<P, enumtype::BSplines, Hexa>;
        //cubature rule
        using cub3=cubature<P, fe3::shape>;
        //finite element discretization
        using fe3_t = intrepid::discretization<fe3, cub3>;
        //![instantiation]
        //finite element
        fe3_t fe3_;



        //![instantiation]

        //computing the value of the basis functions in the quadrature points
        // fe_.compute(Intrepid::OPERATOR_VALUE);
        fe2_.compute(Intrepid::OPERATOR_VALUE);
        fe3_.compute(Intrepid::OPERATOR_VALUE);


        //check correctness:

        std::array<double, 5> knots{-3.,-1.,1.,3., 5.};
        iga_rt::BSpline<1,2> test11(knots);
        iga_rt::BSpline<2,2> test21(knots);
        iga_rt::BSpline<3,2> test31(knots);


        // //partition of unity:
        // std::cout<<"my b spline on "<< fe_.cub_points()(0, 0) <<" : "<<
        //     test11.evaluate(fe_.cub_points()(0, 0)) << "+" <<
        //     test21.evaluate(fe_.cub_points()(0, 0))
        //          <<" = "<<
        //     (test11.evaluate(fe_.cub_points()(0, 0))+
        //      test21.evaluate(fe_.cub_points()(0, 0)))
        //          <<" = "<<fe_.basis_function()(0,0)<<" + "
        //          <<fe_.basis_function()(1,0)
        //          // <<fe_.basis_function()(2,0)
        //          <<std::endl;

        // std::cout<<"my b spline on "<< fe_.cub_points()(0, 1) <<" : "<<
        //     test11.evaluate(fe_.cub_points()(0, 1)) << "+" <<
        //     test21.evaluate(fe_.cub_points()(0, 1))
        //          <<" = "<<
        //     (test11.evaluate(fe_.cub_points()(0, 1))+
        //      test21.evaluate(fe_.cub_points()(0, 1)))
        //          <<" = "<<fe_.basis_function()(0,1)<<" + "
        //          <<fe_.basis_function()(1,1)
        //          // <<fe_.basis_function()(2,0)
        //          <<std::endl;


        // assert(test11.evaluate(fe_.cub_points()(0, 0)) == fe_.basis_function()(0,0));
        // assert(test21.evaluate(fe_.cub_points()(0, 0)) == fe_.basis_function()(1,0));

        // std::cout<<"my b spline: "<<
        //      (
        //          test11.evaluate(0.57735)*test11.evaluate(fe_.cub_points()(0, 1))+
        //          test11.evaluate(0.57735)*test21.evaluate(fe_.cub_points()(0, 1))+
        //          test11.evaluate(0.57735)*test31.evaluate(fe_.cub_points()(0, 1))+
        //          test21.evaluate(0.57735)*test11.evaluate(fe_.cub_points()(0, 1))+
        //          test21.evaluate(0.57735)*test21.evaluate(fe_.cub_points()(0, 1))+
        //          test21.evaluate(0.57735)*test31.evaluate(fe_.cub_points()(0, 1))+
        //          test31.evaluate(0.57735)*test11.evaluate(fe_.cub_points()(0, 1))+
        //          test31.evaluate(0.57735)*test21.evaluate(fe_.cub_points()(0, 1))+
        //          test31.evaluate(0.57735)*test31.evaluate(fe_.cub_points()(0, 1))
        //          )
        //           << std::endl;

        std::cout<<"my 2D b spline on "<< fe2_.cub_points()(0, 0) <<" : "<<
            test11.evaluate(fe2_.cub_points()(0, 0))*test11.evaluate(fe2_.cub_points()(0, 0)) << "+" <<
            test21.evaluate(fe2_.cub_points()(0, 0))*test21.evaluate(fe2_.cub_points()(0, 0))
                 <<" = "<<
            (test11.evaluate(fe2_.cub_points()(0, 0))+
             test21.evaluate(fe2_.cub_points()(0, 0)))
                 <<" = "<<fe2_.val()(0,0)<<" + "
                 <<fe2_.val()(1,0)
            // <<fe_.val()(2,0)
                 <<std::endl;

        assert(test11.evaluate(fe2_.cub_points()(0, 0))*test11.evaluate(fe2_.cub_points()(0, 0)) == fe2_.val()(0,0));
        assert(test11.evaluate(fe2_.cub_points()(0, 0))*test21.evaluate(fe2_.cub_points()(0, 0)) == fe2_.val()(1,0));
        assert(test21.evaluate(fe2_.cub_points()(0, 0))*test11.evaluate(fe2_.cub_points()(0, 0)) == fe2_.val()(2,0));
        assert(test21.evaluate(fe2_.cub_points()(0, 0))*test21.evaluate(fe2_.cub_points()(0, 0)) == fe2_.val()(3,0));



        std::cout<<"my 3D b spline: "<<

            test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))<<" + "<<
            test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))<<" + "<<
            test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))<<" + "<<
            test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))<<" + "<<

            test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))<<" + "<<
            test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))<<" + "<<
            test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))<<" + "<<
            test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))<<" + "

                 <<" = "<<
            (
                test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))+
                test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))+
                test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))+
                test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))+

                test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))+
                test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))+
                test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))+
                test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))+

                test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test31.evaluate(fe3_.cub_points()(0, 0))+
                test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test31.evaluate(fe3_.cub_points()(0, 0))+
                test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test31.evaluate(fe3_.cub_points()(0, 0))+
                test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test31.evaluate(fe3_.cub_points()(0, 0))

                )
                 << std::endl;

        assert(test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0)) == fe3_.val()(0,0));
        assert(test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0)) == fe3_.val()(1,0));
        assert(test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0)) == fe3_.val()(2,0));
        assert(test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0)) == fe3_.val()(3,0));
        assert(test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0)) == fe3_.val()(4,0));
        assert(test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0)) == fe3_.val()(5,0));
        assert(test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0)) == fe3_.val()(6,0));
        assert(test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0)) == fe3_.val()(7,0));

        //assembly
        //geometry definitions
        using geo_map=reference_element<1, Lagrange, Hexa>;
        //geometry discretization
        using geo_cub=cubature<2, geo_map::shape>;

        using geo_t = intrepid::geometry<geo_map, geo_cub>;
        //assembly infrastructure
        using as=assembly<geo_t>;
        using as_base=assembly_base<geo_t>;

        uint_t d1=10;
        uint_t d2=10;
        uint_t d3=10;

        geo_t geo_;
        as assembler(geo_, d1, d2, d3);
        as_base assembler_base(d1,d2,d3);

        geo_.compute(Intrepid::OPERATOR_GRAD);

        using matrix_storage_info_t=storage_info< __COUNTER__,  layout_tt<5> >;
        using matrix_type=storage_t< matrix_storage_info_t >;
        matrix_storage_info_t meta_(d1,d2,d3,fe3::basisCardinality,fe3::basisCardinality);
        matrix_type mass_(meta_, 0.);

        using vector_storage_info_t=storage_info< __COUNTER__,  layout_tt<4> >;
        using vector_type=storage_t< vector_storage_info_t >;
        vector_storage_info_t meta_vec_(d1,d2,d3,fe3::basisCardinality);
        vector_type vector_(meta_vec_, 1.);

        using domain_tuple_t = aggregator_type_tuple< as, as_base>;
        domain_tuple_t domain_tuple_ (assembler, assembler_base);
        using dt = domain_tuple_t;

        typedef arg<dt::size, matrix_type> p_mass;
        typedef arg<dt::size+1, fe3_t::basis_function_storage_t> p_phi;
        typedef arg<dt::size+2, geo_t::grad_storage_t> p_dphi;
        typedef arg<dt::size+3, vector_type> p_vec;

#ifndef __CUDACC__ //surprisingly, this does not compile with nvcc
        auto domain_=domain_tuple_.template domain< p_mass
                                                , p_phi
                                                , p_dphi
                                                , p_vec
                                                >( mass_
                                                   , fe3_.val()
                                                   ,  geo_.grad()
                                                   , vector_
                                                    );
#else
        aggregator_type<boost::mpl::vector
                    <dt::super::p_grid_points
                     ,dt::p_jac, dt::p_weights, dt::p_jac_det, dt::p_jac_inv,
                     p_mass , p_phi, p_dphi, p_vec> >
            domain_
            ( boost::fusion::make_vector(&assembler.grid()
                                         , &assembler.jac(), &assembler.cub_weights(), &assembler.jac_det(), &assembler.jac_inv()
                                         ,  &mass_ , &fe3_.val(),  &geo_.grad(), &vector_));
#endif

        auto coords=grid<axis>({1, 0, 1, d1-1, d1},
            {1, 0, 1, d2-1, d2});
        coords.value_list[0] = 0;
        coords.value_list[1] = d3-1;

        auto computation=make_computation<gridtools::BACKEND>(
            make_multistage
            (
                execute<forward>()
                , make_stage<functors::update_jac<geo_t> >( dt::p_grid_points(), p_dphi(), dt::p_jac())
                , make_stage<functors::det<geo_t> >(dt::p_jac(), dt::p_jac_det())
                , make_stage<functors::mass<fe3, geo_cub> >(dt::p_jac_det(), dt::p_weights(), p_phi(), p_phi(), p_mass())
                , make_stage<functors::matvec>(p_vec(), p_mass(), p_vec())//matrix vector product
                //, make_stage<functors::jump_f<geo_t> >(p_vec(), p_vec())//compute the jump
                ), domain_, coords);

        computation->ready();
        computation->steady();
        computation->run();
        computation->finalize();
        vector_.print();
    }//namespace mass_IGA
}//namespace functors

/**
\file
*/
#pragma once
#define PEDANTIC_DISABLED
#include "assembly.h"

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
    fe_.compute(Intrepid::OPERATOR_VALUE);
    fe2_.compute(Intrepid::OPERATOR_VALUE);
    fe3_.compute(Intrepid::OPERATOR_VALUE);



    //check correctness:

    gridtools::array<double, 5> knots{-3.,-1.,1.,3., 5.};
    iga_rt::BSpline<1,2> test11(&knots);
    iga_rt::BSpline<2,2> test21(&knots);
    iga_rt::BSpline<3,2> test31(&knots);


    //partition of unity:
    std::cout<<"my b spline on "<< fe_.cub_points()(0, 0) <<" : "<<
        test11.evaluate(fe_.cub_points()(0, 0)) << "+" <<
        test21.evaluate(fe_.cub_points()(0, 0))
             <<" = "<<
        (test11.evaluate(fe_.cub_points()(0, 0))+
         test21.evaluate(fe_.cub_points()(0, 0)))
             <<" = "<<fe_.basis_function()(0,0)<<" + "
             <<fe_.basis_function()(1,0)
             // <<fe_.basis_function()(2,0)
             <<std::endl;

    std::cout<<"my b spline on "<< fe_.cub_points()(0, 1) <<" : "<<
        test11.evaluate(fe_.cub_points()(0, 1)) << "+" <<
        test21.evaluate(fe_.cub_points()(0, 1))
             <<" = "<<
        (test11.evaluate(fe_.cub_points()(0, 1))+
         test21.evaluate(fe_.cub_points()(0, 1)))
             <<" = "<<fe_.basis_function()(0,1)<<" + "
             <<fe_.basis_function()(1,1)
             // <<fe_.basis_function()(2,0)
             <<std::endl;


    assert(test11.evaluate(fe_.cub_points()(0, 0)) == fe_.basis_function()(0,0));
    assert(test21.evaluate(fe_.cub_points()(0, 0)) == fe_.basis_function()(1,0));

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
             <<" = "<<fe2_.basis_function()(0,0)<<" + "
             <<fe2_.basis_function()(1,0)
             // <<fe_.basis_function()(2,0)
             <<std::endl;

    assert(test11.evaluate(fe2_.cub_points()(0, 0))*test11.evaluate(fe2_.cub_points()(0, 0)) == fe2_.basis_function()(0,0));
    assert(test11.evaluate(fe2_.cub_points()(0, 0))*test21.evaluate(fe2_.cub_points()(0, 0)) == fe2_.basis_function()(1,0));
    assert(test21.evaluate(fe2_.cub_points()(0, 0))*test11.evaluate(fe2_.cub_points()(0, 0)) == fe2_.basis_function()(2,0));
    assert(test21.evaluate(fe2_.cub_points()(0, 0))*test21.evaluate(fe2_.cub_points()(0, 0)) == fe2_.basis_function()(3,0));



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

     assert(test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0)) == fe3_.basis_function()(0,0));
     assert(test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0)) == fe3_.basis_function()(1,0));
     assert(test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0)) == fe3_.basis_function()(2,0));
     assert(test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0)) == fe3_.basis_function()(3,0));
     assert(test11.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0)) == fe3_.basis_function()(4,0));
     assert(test21.evaluate(fe3_.cub_points()(0, 0))*test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0)) == fe3_.basis_function()(5,0));
     assert(test11.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0)) == fe3_.basis_function()(6,0));
     assert(test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0))*test21.evaluate(fe3_.cub_points()(0, 0)) == fe3_.basis_function()(7,0));

     //assembly
     //geometry definitions
     using geo_map=reference_element<1, Lagrange, Hexa>;
     //geometry discretization
     using geo_t = intrepid::geometry<geo_map, cub>;
     //assembly infrastructure
     using as=assembly<geo_t>;

     //    geo_t geo_;
     //    as assembler(geo_, P+1, P+1, P+1);

}

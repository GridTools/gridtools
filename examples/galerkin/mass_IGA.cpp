/**
\file
*/
#pragma once
#define PEDANTIC_DISABLED
#include "assembly.h"

int main(){

    static const int P=2;
    //the finite elements definitions
    using fe=reference_element<P, enumtype::BSplines, Hexa>;
    //geometry definitions
    using geo_map=reference_element<1, Lagrange, Hexa>;
    //cubature rule
    using cub=cubature<P, fe::shape>;
    //finite element discretization
    using fe_t = intrepid::discretization<fe, cub>;
    //geometry discretization
    using geo_t = intrepid::geometry<geo_map, cub>;
    //assembly infrastructure
    using as=assembly<geo_t>;

    //![instantiation]
    //finite element
    fe_t fe_;
    //geometry
    geo_t geo_;


    as assembler(geo_, P+1, P+1, P+1);
    //![instantiation]

    //computing the value of the basis functions in the quadrature points
    fe_.compute(Intrepid::OPERATOR_VALUE);

    for(int i=0; i<fe_.basis_function().meta_data().dims<1>(); ++i){
        for(int j=0; j<fe_.basis_function().meta_data().dims<0>(); ++j){
            std::cout<<fe_.basis_function()(j,i,0)<<" ";
        }
        std::cout<<std::endl;
    }


    gridtools::array<double, 3> knots{0,1,2};
    iga_rt::BSpline<1,1> test(&knots);
    std::cout<<"my b spline"<<test.evaluate(0.5)<<std::endl;

}

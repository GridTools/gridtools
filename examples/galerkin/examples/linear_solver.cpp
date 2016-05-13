// TODO: clean include list, assembly.hpp should not be here
#include <iostream>
#include "../numerics/assembly.hpp"
#include "../numerics/assemble_storage.hpp"
#include "../numerics/linear_solver.hpp"

using namespace gridtools;
using namespace gdl;
using namespace gdl::enumtype;

typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;

constexpr unsigned int dof_per_dim_0(4);
constexpr unsigned int dof_per_dim_1(4);
constexpr unsigned int dof_per_dim_2(4);

constexpr unsigned int d1=4;
constexpr unsigned int d2=4;
constexpr unsigned int d3=4;
constexpr unsigned int d4=dof_per_dim_0*dof_per_dim_1*dof_per_dim_2;

constexpr gridtools::uint_t n_dof{((dof_per_dim_0-1)*d1+1)*((dof_per_dim_1-1)*d2+1)*((dof_per_dim_2-1)*d3+1)};

int main() {

    // - In this example a linear problem Ax=b is solved using conjugate gradient method
    // - A and b vector are unassembled while the solution x will be provided in assembled
    // form and when a non zero starting point is provided, in the same variable, it must
    // be in its assembled form
    // - "left" and "right" halos are expected to be present in the provided data: if
    // d1, d2, d3 and d4 are the provided storage sizes, the problem is solved for the
    // "physical" central d1-2, d2-2, d3-2 and d4-2 elements
    // - A and b elements for halo element must be set to zero
    // TODO: last two condition must be removed from external solver interface

    //![solver_setup]
    const double stability_thr(-1.0);
    const double error_thr(1.e-6);
    double stability;
    uint_t max_iter(50);
    //![solver_setup]

    //![storages]
    // Problem matrix
    using matrix_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4> >;
    using matrix_type=storage_t< matrix_storage_info_t >;
    matrix_storage_info_t A_(d1,d2,d3,d4,d4);
    matrix_type A(A_, 0.e0, "A");// This is the unassembled problem matrix

    // RHS vector
    using rhs_vector_storage_info_t=storage_info< __COUNTER__, layout_tt<3> >;
    using rhs_vector_type=storage_t< rhs_vector_storage_info_t >;
    rhs_vector_storage_info_t b_(d1,d2,d3,d4);
    rhs_vector_type b(b_, 0.e0, "b");// This is the unassembled right hand side vector

    // Unknowns vector
    using unk_vector_storage_info_t=storage_info< __COUNTER__, layout_tt<3> >;
    using unk_vector_type = storage<assemble_storage< unk_vector_storage_info_t, dof_per_dim_0, dof_per_dim_1, dof_per_dim_2> >;
    unk_vector_storage_info_t x_(d1,d2,d3,d4);
    unk_vector_type x(x_, 0.e0, "x");// This is the assembled unknowns vector
    //![storages]


    //![input_data]
    for(uint_t I=1; I<d1-1; I++)
        for(uint_t J=1; J<d2-1; J++)
            for(uint_t K=1; K<d3-1; K++)
                for(uint_t dof_index1=0; dof_index1<d4; dof_index1++)
                    for(uint_t dof_index2=0; dof_index2<d4; dof_index2++)
                    {
                        if(dof_index1 == dof_index2){
                            if(dof_index1 == 0)
                                A(I,J,K,dof_index1,dof_index2) = 2.e0;
                            else
                                A(I,J,K,dof_index1,dof_index2) = 1.e0;
                        }
                        else
                            A(I,J,K,dof_index1,dof_index2) = 0.e0;
                    }

    for(uint_t I=1; I<d1-1; I++)
        for(uint_t J=1; J<d2-1; J++)
            for(uint_t K=1; K<d3-1; K++)
                for(uint_t dof_index=0; dof_index<d4; dof_index++)
                {
                    if(dof_index == 0 || dof_index==1)
                        b(I,J,K,dof_index) = 1.e0;
                }
    //![input_data]


    //![solve]
    linear_solver< gdl::cg_solver<dof_per_dim_0,dof_per_dim_1,dof_per_dim_2> >::solve(A, b, x, stability_thr, error_thr, max_iter);
    //![solve]


    //![print_result]
    // Print non-zero results
    for(uint_t dof = 0;dof<n_dof;++dof) {

        if(x.get_value(dof))
            std::cout<<dof<<" "<<x.get_value(dof)<<std::endl;

    }
    //![print_result]

}

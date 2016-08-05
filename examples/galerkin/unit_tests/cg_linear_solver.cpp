// TODO: clean include list, assembly.hpp should not be here
#include <iostream>
#include <iomanip>
#include <random>
#include <fstream>
#include "../numerics/assembly.hpp"
#include "../numerics/assemble_storage.hpp"
#include "../numerics/linear_solver.hpp"

using namespace gridtools;
using namespace gdl;
using namespace gdl::enumtype;

constexpr unsigned int dof_per_dim_0(3);
constexpr unsigned int dof_per_dim_1(3);
constexpr unsigned int dof_per_dim_2(3);

constexpr unsigned int d1=5;
constexpr unsigned int d2=5;
constexpr unsigned int d3=5;
constexpr unsigned int d4=dof_per_dim_0*dof_per_dim_1*dof_per_dim_2;

constexpr gridtools::uint_t n_dof{((dof_per_dim_0-1)*(d1-2)+1)*((dof_per_dim_1-1)*(d2-2)+1)*((dof_per_dim_2-1)*(d3-2)+1)};

#define PRINT_DATA_FILES true

int main() {

    // - In this example a linear problem Ax=b is solved using conjugate gradient method
    // - A and b vector are unassembled while the solution x will be provided in assembled
    // form and when a non zero starting point is provided, in the same variable, it must
    // be in its assembled form
    // - The A matrix is randomly generated as A=Z*Z' where Z is in its turn randomly generated,
    // this ensure A to be invertible. b vector is randomly generated too as b = Ax where the
    // problem solution x is also random.
    // - "left" and "right" halos are expected to be present in the provided data: if
    // d1, d2, d3 and d4 are the provided storage sizes, the problem is solved for the
    // "physical" central d1-2, d2-2, d3-2 and d4-2 elements
    // - A and b elements for halo element must be set to zero
    // - Setting PRINT_DATA_FILES to true, files containing the assembled A,b and x data
    //   are produced. They can be used for cross check (e.g., vs python numpy solution)
    // TODO: last two condition must be removed from external solver interface

    //![test_check_values]
    const float_t cumulative_error_check{1e-7};
    const float_t max_relative_error_check{0.5};// This is in %
    //![test_check_values]

    //![solver_setup]
    const double stability_thr(-1.0);
    const double error_thr(1.e-20);
    double stability;
    uint_t max_iter(20000);
    //![solver_setup]


    //![storages]
    // Problem matrix
    using matrix_storage_info_t=storage_info< __COUNTER__, layout_tt<5> >;
    using matrix_type=storage_t< matrix_storage_info_t >;
    matrix_storage_info_t A_(d1,d2,d3,d4,d4);
    matrix_type A(A_, 0.e0, "A");// This is the unassembled problem matrix

    // RHS vector
    using rhs_vector_storage_info_t=storage_info< __COUNTER__, layout_tt<4> >;
    using rhs_vector_type=storage_t< rhs_vector_storage_info_t >;
    rhs_vector_storage_info_t b_(d1,d2,d3,d4);
    rhs_vector_type b(b_, 0.e0, "b");// This is the unassembled right hand side vector

    // Unknowns vector
    using unk_vector_storage_info_t=storage_info< __COUNTER__, layout_tt<4> >;
    using unk_vector_type = storage<assemble_storage< unk_vector_storage_info_t, dof_per_dim_0, dof_per_dim_1, dof_per_dim_2> >;
    unk_vector_storage_info_t x_(d1,d2,d3,d4);
    unk_vector_type x(x_, halo_data(2,2,2,1,1,1), 0.e0, "x");// This is the assembled unknowns vector
    //![storages]

    //![random_generator_setup]
    std::default_random_engine rand_eng;
    std::uniform_real_distribution<float_t> rand_dist(0.e0,1.e0);
    //![random_generator_setup]

    //![random_problem_generation]
    std::vector<float_t> A_rand(n_dof*n_dof);
    std::vector<float_t> A_ext(n_dof*n_dof);
    std::vector<float_t> x_ext(n_dof);
    std::vector<float_t> b_ext(n_dof);

    // Generate random matrix
    for(uint_t row=0;row<n_dof;++row)
        for(uint_t col=0;col<n_dof;++col)
            A_rand[row*n_dof+col] = rand_dist(rand_eng);

    // Compute linear system matrix A_ext as A_rand*A_rand^T
    for(uint_t row=0;row<n_dof;++row)
        for(uint_t col=0;col<n_dof;++col){
            float_t sum = 0.;
            for(uint_t index=0;index<n_dof;++index)
                sum += A_rand[row*n_dof+index]*A_rand[col*n_dof+index];
            A_ext[row*n_dof+col] = sum;
        }

    // Generate random solution
    for(uint_t row=0;row<n_dof;++row)
        x_ext[row] = rand_dist(rand_eng);

    // Compute adjacency data (number of contributors for matrix each element)
    std::vector<int> number_contributions_A(n_dof*n_dof,0);
    std::vector<int> number_contributions_b(n_dof,0);
    constexpr uint_t dof_per_dim(dof_per_dim_0);
    constexpr uint_t d1_2{d1-2};
    constexpr uint_t d2_2{d2-2};
    constexpr uint_t d3_2{d3-2};
    for (uint_t i=0; i<d1_2; i++)
        for (uint_t j=0; j<d2_2; j++)
            for (uint_t k=0; k<d3_2; k++)
                for(uint_t dofx1 = 0;dofx1<dof_per_dim;++dofx1)
                    for(uint_t dofy1 = 0;dofy1<dof_per_dim;++dofy1)
                        for(uint_t dofz1 = 0;dofz1<dof_per_dim;++dofz1)
                            for(uint_t dofx2 = 0;dofx2<dof_per_dim;++dofx2)
                                for(uint_t dofy2 = 0;dofy2<dof_per_dim;++dofy2)
                                    for(uint_t dofz2 = 0;dofz2<dof_per_dim;++dofz2){

                                        assert((
                                                i*(dof_per_dim-1) + dofx1 +
                                                j*((dof_per_dim-1)*d1_2 + 1)*(dof_per_dim-1) + dofy1*((dof_per_dim-1)*d1_2 + 1) +
                                                k*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1)*(dof_per_dim-1) + dofz1*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1)
                                                )*n_dof
                                                +
                                                i*(dof_per_dim-1) + dofx2 +
                                                j*((dof_per_dim-1)*d1_2 + 1)*(dof_per_dim-1) + dofy2*((dof_per_dim-1)*d1_2 + 1) +
                                                k*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1)*(dof_per_dim-1) + dofz2*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1)<n_dof*n_dof);

                                        number_contributions_A[
                                            (
                                            i*(dof_per_dim-1) + dofx1 +
                                            j*((dof_per_dim-1)*d1_2 + 1)*(dof_per_dim-1) + dofy1*((dof_per_dim-1)*d1_2 + 1) +
                                            k*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1)*(dof_per_dim-1) + dofz1*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1)
                                            )*n_dof
                                            +
                                            i*(dof_per_dim-1) + dofx2 +
                                            j*((dof_per_dim-1)*d1_2 + 1)*(dof_per_dim-1) + dofy2*((dof_per_dim-1)*d1_2 + 1) +
                                            k*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1)*(dof_per_dim-1) + dofz2*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1)]++;
                                    }

    for (uint_t i=0; i<d1_2; i++)
        for (uint_t j=0; j<d2_2; j++)
            for (uint_t k=0; k<d3_2; k++)
                for(uint_t dofx1 = 0;dofx1<dof_per_dim;++dofx1)
                    for(uint_t dofy1 = 0;dofy1<dof_per_dim;++dofy1)
                        for(uint_t dofz1 = 0;dofz1<dof_per_dim;++dofz1){

                                assert(
                                        i*(dof_per_dim-1) + dofx1 +
                                        j*((dof_per_dim-1)*d1_2 + 1)*(dof_per_dim-1) + dofy1*((dof_per_dim-1)*d1_2 + 1) +
                                        k*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1)*(dof_per_dim-1) + dofz1*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1)<n_dof);

                                number_contributions_b[
                                    i*(dof_per_dim-1) + dofx1 +
                                    j*((dof_per_dim-1)*d1_2 + 1)*(dof_per_dim-1) + dofy1*((dof_per_dim-1)*d1_2 + 1) +
                                    k*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1)*(dof_per_dim-1) + dofz1*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1)
                                    ]++;
                            }


    // Build unassembled A matrix (shared contributions are split into equal parts)
    for(uint_t i=0; i<d1_2; i++)
        for(uint_t j=0; j<d2_2; j++)
            for(uint_t k=0; k<d3_2; k++)
                for(uint_t dofx1 = 0;dofx1<dof_per_dim;++dofx1)
                    for(uint_t dofy1 = 0;dofy1<dof_per_dim;++dofy1)
                        for(uint_t dofz1 = 0;dofz1<dof_per_dim;++dofz1){
                            const uint_t dof1 = dofx1 + dofy1*dof_per_dim + dofz1*dof_per_dim*dof_per_dim;
                            for(uint_t dofx2 = 0;dofx2<dof_per_dim;++dofx2)
                                for(uint_t dofy2 = 0;dofy2<dof_per_dim;++dofy2)
                                    for(uint_t dofz2 = 0;dofz2<dof_per_dim;++dofz2)
                                    {
                                        const uint_t dof2 = dofx2 + dofy2*dof_per_dim + dofz2*dof_per_dim*dof_per_dim;
                                        const uint_t local_to_global =
                                                (
                                                i*(dof_per_dim-1) + dofx1 +
                                                j*((dof_per_dim-1)*d1_2 + 1)*(dof_per_dim-1) + dofy1*((dof_per_dim-1)*d1_2 + 1) +
                                                k*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1)*(dof_per_dim-1) + dofz1*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1)
                                                )*n_dof
                                                +
                                                i*(dof_per_dim-1) + dofx2 +
                                                j*((dof_per_dim-1)*d1_2 + 1)*(dof_per_dim-1) + dofy2*((dof_per_dim-1)*d1_2 + 1) +
                                                k*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1)*(dof_per_dim-1) + dofz2*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1);

                                        assert(local_to_global<n_dof*n_dof);

                                        A(i+1,j+1,k+1,dof1,dof2) = A_ext[local_to_global]/number_contributions_A[local_to_global];
                                    }
                        }

    // Reset unwanted (non-FEM) matrix element: these are the contributions
    // that would come from functions belonging to different mesh elements
    for(uint_t row=0;row<n_dof;++row)
        for(uint_t col=0;col<n_dof;++col)
            if(number_contributions_A[row*n_dof+col]==0)
                A_ext[row*n_dof+col] = 0.;


    // Compute random RHS
    for(uint_t row=0;row<n_dof;++row){
        float_t sum = 0.;
        for(uint_t index=0;index<n_dof;++index)
            sum += A_ext[row*n_dof+index]*x_ext[index];
        b_ext[row] = sum;
    }


    // Build unassembled b vector (shared contributions are split into equal parts)
    for(uint_t i=0; i<d1_2; i++)
        for(uint_t j=0; j<d2_2; j++)
            for(uint_t k=0; k<d3_2; k++)
                for(uint_t dofx1 = 0;dofx1<dof_per_dim;++dofx1)
                    for(uint_t dofy1 = 0;dofy1<dof_per_dim;++dofy1)
                        for(uint_t dofz1 = 0;dofz1<dof_per_dim;++dofz1)
                        {
                            const uint_t dof1 = dofx1 + dofy1*dof_per_dim + dofz1*dof_per_dim*dof_per_dim;
                            const uint_t local_to_global =
                                    i*(dof_per_dim-1) + dofx1 +
                                    j*((dof_per_dim-1)*d1_2 + 1)*(dof_per_dim-1) + dofy1*((dof_per_dim-1)*d1_2 + 1) +
                                    k*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1)*(dof_per_dim-1) + dofz1*((dof_per_dim-1)*d1_2 + 1)*((dof_per_dim-1)*d2_2 + 1);
                            assert(local_to_global<n_dof);
                            b(i+1,j+1,k+1,dof1) = b_ext[local_to_global]/number_contributions_b[local_to_global];
                        }

    //![random_problem_generation]

    //![solve]
    linear_solver< gdl::cg_solver<dof_per_dim_0,dof_per_dim_1,dof_per_dim_2> >::solve(A, b, x, stability_thr, error_thr, max_iter);
    //![solve]

    //![check_results]
    bool success = true;
    float_t cumulative_error=0;
    float_t max_relative_error=0;
    for(uint_t dof = 0;dof<n_dof;++dof) {
        const float_t cg_x(x.storage_pointer()->get_value(dof));
        float_t relative_error(cg_x-x_ext[dof]);
        cumulative_error += relative_error*relative_error;
        if(x_ext[dof]!=0){
            relative_error /= x_ext[dof];
            if(std::abs(relative_error>max_relative_error))
                max_relative_error = relative_error;
        }
    }

    std::cout<<"Cumulative error = "<<cumulative_error<<std::endl;
    std::cout<<"Max relative (abs) error = "<<max_relative_error*100<<std::endl;

    if(cumulative_error>cumulative_error_check) {
        success = false;
        std::cout<<"Cumulative error check not passed, check value set to "<<cumulative_error_check<<std::endl;
    }

    if(max_relative_error>max_relative_error_check) {
        success = false;
        std::cout<<"Max relative error check not passed, check value set to "<<max_relative_error_check<<std::endl;
    }
    //![check_results]


#if PRINT_DATA_FILES
    //![print_result]
    std::ofstream A_matrix;
    A_matrix.open("A_matrix.dat");
    A_matrix<<std::setprecision(10);
    for(uint_t dof1 = 0;dof1<n_dof;++dof1){
         for(uint_t dof2 = 0;dof2<n_dof;++dof2)
             A_matrix<<A_ext[dof1*n_dof+dof2]<<" ";
    }
    A_matrix.close();

    std::ofstream b_vector;
    b_vector.open("b_vector.dat");
    b_vector<<std::setprecision(10);
    for(uint_t dof1 = 0;dof1<n_dof;++dof1){
            b_vector<<b_ext[dof1]<<" ";
    }
    b_vector.close();

    std::ofstream x_vector;
    x_vector.open("x_vector.dat");
    x_vector<<std::setprecision(10);
    for(uint_t dof1 = 0;dof1<n_dof;++dof1){
        x_vector<<x.storage_pointer()->get_value(dof1)<<" ";
    }
    x_vector.close();

    //![print_result]
#endif

    assert(success);
}

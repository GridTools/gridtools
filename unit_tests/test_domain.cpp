/* 
 * File:   test_domain.cpp
 * Author: mbianco
 *
 * Created on February 5, 2014, 4:16 PM
 * 
 * Test domain features, especially the working on the GPU
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <gpu_clone.h>
#include <hybrid_pointer.h>
#include <cuda_storage.h>
#include <domain_type.h>
#include <arg_type.h>

struct out_value {
    template <typename T>
    __device__
    void operator()(T const& x) const {
        printf("%X\n", x);
        // for (int i=0; i<3; ++i) {
        //     for (int j=0; j<3; ++j) {
        //         for (int k=0; k<3; ++k) {
        //             printf("%d ", (*x)(i,j,k));
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }
    }
};

template <typename t_domain>
__global__
void print_values(t_domain const* domain) {
    boost::fusion::for_each(domain->args, out_value());
}

/*
 * 
 */
int main(int argc, char** argv) {

    typedef gridtools::cuda_storage<double, gridtools::layout_map<0,1,2> > storage_type;

    int d1 = atoi(argv[1]);
    int d2 = atoi(argv[2]);
    int d3 = atoi(argv[3]);
    
    storage_type in(d1,d2,d3,-1, std::string("in"));
    storage_type out(d1,d2,d3,-7.3, std::string("out"));
    storage_type coeff(d1,d2,d3,-3, std::string("coeff"));

    // Definition of placeholders. The order of them reflect the order the user will deal with them
    // especially the non-temporary ones, in the construction of the domain
    typedef gridtools::arg<3, gridtools::temporary<storage_type> > p_lap;
    typedef gridtools::arg<4, gridtools::temporary<storage_type> > p_flx;
    typedef gridtools::arg<5, gridtools::temporary<storage_type> > p_fly;
    typedef gridtools::arg<0, storage_type > p_coeff;
    typedef gridtools::arg<1, storage_type > p_in;
    typedef gridtools::arg<2, storage_type > p_out;

    // An array of placeholders to be passed to the domain
    // I'm using mpl::vector, but the final API should look slightly simpler
    typedef boost::mpl::vector</*p_lap, p_flx, p_fly*/ p_coeff, p_in, p_out> arg_type_list;

    // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
    // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
    gridtools::domain_type<arg_type_list> domain
        (boost::fusion::make_vector(&coeff, &in, &out /*,&fly, &flx*/));

    domain.clone_to_gpu();

    print_values<<<1,1>>>(domain.gpu_object_ptr);

    std::cout << " *** DONE ***" << std::endl;

    return 0;
}


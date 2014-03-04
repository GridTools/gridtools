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
    __host__ __device__
    void operator()(T *x) const {
#ifndef NDEBUG
        printf("gigigi ");
        printf("%X\n", x->data.pointer_to_use);
        printf("%X\n", x->data.cpu_p);
        printf("%X\n", x->data.gpu_p);
        printf("%d\n", x->data.size);
#endif
        for (int i=0; i<3; ++i) {
            for (int j=0; j<3; ++j) {
                for (int k=0; k<3; ++k) {
#ifndef NDEBUG
                    printf("%e ", (*x)(i,j,k));
#endif
                    (*x)(i,j,k) = 1+2*((*x)(i,j,k));
#ifndef NDEBUG
                    printf("%e ", (*x)(i,j,k));
                    printf("\n");
#endif
                }
            }
        }
    }
};

struct out_value_ {
    template <typename T>
    __host__ __device__
    void operator()(T const& stor) const {
        //std::cout << __PRETTY_FUNCTION__ << std::endl;
        printf(" > %X %X\n", &stor, stor.data.pointer_to_use);
        for (int i=0; i<3; ++i) {
            for (int j=0; j<3; ++j) {
                for (int k=0; k<3; ++k) {
                    printf("%e, ", stor(i,j,k));
                }
                printf("\n");
            }
            printf("\n");
        }
    }
};

template <typename t_domain>
__global__
void print_values(t_domain const* domain) {
    boost::fusion::for_each(domain->storage_pointers, out_value());
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
    storage_type coeff(d1,d2,d3,-3.4, std::string("coeff"));

    // Definition of placeholders. The order of them reflect the order the user will deal with them
    // especially the non-temporary ones, in the construction of the domain
    // typedef gridtools::arg<3, gridtools::temporary<storage_type> > p_lap;
    // typedef gridtools::arg<4, gridtools::temporary<storage_type> > p_flx;
    // typedef gridtools::arg<5, gridtools::temporary<storage_type> > p_fly;
    typedef gridtools::arg<0, storage_type > p_coeff;
    typedef gridtools::arg<1, storage_type > p_in;
    typedef gridtools::arg<2, storage_type > p_out;


    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d2; ++j) {
            for (int k = 0; k < d3; ++k) {
                coeff(i,j,k) = -1*(i+j+k)*3.4;
                //std::cout << coeff(i,j,k) << " ";
                out(i,j,k) = -1*(i+j+k)*100;
                //std::cout << out(i,j,k) << " ";
                in(i,j,k) = -1*(i+j+k)*0.45;
                //std::cout << in(i,j,k) << " ";
            }
        }
    }

    // // An array of placeholders to be passed to the domain
    // // I'm using mpl::vector, but the final API should look slightly simpler
    typedef boost::mpl::vector</*p_lap, p_flx, p_fly*/ p_coeff, p_in, p_out> arg_type_list;

    // // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
    // // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
    gridtools::domain_type<arg_type_list> domain
        (boost::fusion::make_vector(&coeff, &in, &out /*,&fly, &flx*/));


    printf(" > %X %X\n", &coeff, coeff.data.pointer_to_use);
    out_value_()(coeff);
    printf(" > %X %X\n", &in, in.data.pointer_to_use);
    out_value_()(in);
    printf(" > %X %X\n", &out, out.data.pointer_to_use);
    out_value_()(out);

    // THERE ARE NOT TEMPS HERE    domain.prepare_temporaries();
    domain.is_ready=true;
    domain.setup_computation();
    domain.clone_to_gpu();

    printf("\n\nFROM GPU\n\n");
    print_values<<<1,1>>>(domain.gpu_object_ptr);
    cudaDeviceSynchronize();
    printf("\n\nDONE WITH GPU\n\n");

    domain.finalize_computation();

    coeff.data.update_cpu();
    in.data.update_cpu();
    out.data.update_cpu();

    printf(" > %X %X\n", &coeff, coeff.data.pointer_to_use);
    out_value_()(coeff);
    printf(" > %X %X\n", &in, in.data.pointer_to_use);
    out_value_()(in);
    printf(" > %X %X\n", &out, out.data.pointer_to_use);
    out_value_()(out);

    std::cout << "\n\n\nTEST 2\n\n\n" << std::endl;

    domain.setup_computation();
    domain.clone_to_gpu();

    printf("\n\nFROM GPU\n\n");
    print_values<<<1,1>>>(domain.gpu_object_ptr);
    cudaDeviceSynchronize();
    printf("\n\nDONE WITH GPU\n\n");

    domain.finalize_computation();

    coeff.data.update_cpu();
    in.data.update_cpu();
    out.data.update_cpu();

    printf(" > %X %X\n", &coeff, coeff.data.pointer_to_use);
    out_value_()(coeff);
    printf(" > %X %X\n", &in, in.data.pointer_to_use);
    out_value_()(in);
    printf(" > %X %X\n", &out, out.data.pointer_to_use);
    out_value_()(out);

    std::cout << " *** DONE ***" << std::endl;

    return 0;
}


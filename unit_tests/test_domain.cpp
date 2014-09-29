/*
 * File:   test_domain.cpp
 * Author: mbianco
 *
 * Created on February 5, 2014, 4:16 PM
 *
 * Test domain features, especially the working on the GPU
 */

#define BOOST_NO_CXX11_RVALUE_REFERENCES

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <common/gpu_clone.h>
#include <storage/hybrid_pointer.h>
#include <storage/cuda_storage.h>
#include <stencil-composition/domain_type.h>
#include <stencil-composition/arg_type.h>
#include <stencil-composition/intermediate.h>


#include <boost/current_function.hpp>
#include <boost/fusion/include/nview.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/mpl/vector.hpp>

struct out_value {
    template <typename T>
    __host__ __device__
    void operator()(T *x) const {
#ifndef NDEBUG
        printf("gigigi ");
        printf("%X\n", x->m_data.get_pointer_to_use());
        printf("%X\n", x->m_data.get_cpu_p());
        printf("%X\n", x->m_data.get_gpu_p());
        printf("%d\n", x->m_data.get_size());
#endif
        for (int i=0; i<3; ++i) {
            for (int j=0; j<3; ++j) {
                for (int k=0; k<3; ++k) {
#ifndef NDEBUG
                    printf("%e ", (*x)(i,j,k));
#endif
                    (*x)(i,j,k) = 1+2*((*x)(i,j,k));
#ifndef NDEBUG
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 3200)
                    printf("GPU 1+2* that is: %e ", (*x)(i,j,k));
#else
                    printf("CPU 1+2* that is: %e ", (*x)(i,j,k));
#endif
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
        //std::cout << BOOST_CURRENT_FUNCTION << std::endl;
        //printf(" > %X %X\n", &stor, stor.m_data.get_pointer_to_use());
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

template <typename StoragePtrs>
__global__
void print_values(StoragePtrs const* storage_pointers) {
    boost::fusion::for_each(*storage_pointers, out_value());
}

template <typename One, typename Two>
bool the_same(One const& storage1, Two const& storage2) {
    bool same = true;
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            for (int k=0; k<3; ++k) {
                same &= (storage1(i,j,k) == storage2(i,j,k));
                if ((storage1(i,j,k) != storage2(i,j,k))) {
                    std::cout << i << ", "
                              << j << ", "
                              << k << ": "
                              << storage1(i,j,k) << " != "
                              << storage2(i,j,k)
                              << std::endl;
                }
            }
        }
    }
    return same;
}

/*
 *
 */
bool test_domain() {

    typedef gridtools::base_storage<gridtools::enumtype::Cuda, double, gridtools::layout_map<0,1,2> > storage_type;

    int d1 = 3;
    int d2 = 3;
    int d3 = 3;

    storage_type in(d1,d2,d3,-1, std::string("in"));
    storage_type out(d1,d2,d3,-7.3, std::string("out"));
    storage_type coeff(d1,d2,d3,-3.4, std::string("coeff"));

    storage_type host_in(d1,d2,d3,-1, std::string("host_in"));
    storage_type host_out(d1,d2,d3,-7.3, std::string("host_out"));
    storage_type host_coeff(d1,d2,d3,-3.4, std::string("host_coeff"));

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
                out(i,j,k) = -1*(i+j+k)*100;
                in(i,j,k) = -1*(i+j+k)*0.45;
                host_coeff(i,j,k) = -1*(i+j+k)*3.4;
                host_out(i,j,k) = -1*(i+j+k)*100;
                host_in(i,j,k) = -1*(i+j+k)*0.45;
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


#ifndef NDEBUG
    printf("coeff > %X %X\n", &coeff, coeff.m_data.get_pointer_to_use());
    out_value_()(coeff);
    printf("in    > %X %X\n", &in, in.m_data.get_pointer_to_use());
    out_value_()(in);
    printf("out   > %X %X\n", &out, out.m_data.get_pointer_to_use());
    out_value_()(out);
#endif

    typedef boost::mpl::vector<
  gridtools::_impl::select_storage<arg_type_list>::template apply<boost::mpl::int_<0> >::type,
    gridtools::_impl::select_storage<arg_type_list>::template apply<boost::mpl::int_<1> >::type,
    gridtools::_impl::select_storage<arg_type_list>::template apply<boost::mpl::int_<2> >::type
    > mpl_arg_type_list;

    typedef typename boost::fusion::result_of::as_vector<mpl_arg_type_list>::type actual_arg_list_type;

    actual_arg_list_type actual_arg_list;

    boost::fusion::copy(domain.storage_pointers, actual_arg_list);

#ifdef __CUDACC__
    gridtools::setup_computation<gridtools::enumtype::Cuda>::apply( actual_arg_list, domain );
#else
    gridtools::setup_computation<gridtools::enumtype::Host>::apply( actual_arg_list, domain ); //does nothing
#endif
    
    actual_arg_list_type* arg_list_device_ptr;
    cudaMalloc(&arg_list_device_ptr, sizeof(actual_arg_list_type));
    cudaMemcpy(arg_list_device_ptr, &actual_arg_list , sizeof(actual_arg_list_type), cudaMemcpyHostToDevice);

#ifndef NDEBUG
    printf("\n\nFROM GPU\n\n");
#endif
    print_values<<<1,1>>>(arg_list_device_ptr/*domain.gpu_object_ptr*/);
    cudaDeviceSynchronize();
#ifndef NDEBUG
    printf("\n\nDONE WITH GPU\n\n");
#endif
    domain.finalize_computation();

    coeff.m_data.update_cpu();
    in.m_data.update_cpu();
    out.m_data.update_cpu();

#ifndef NDEBUG
    printf("back coeff > %X %X\n", coeff.m_data.get_cpu_p(), coeff.m_data.get_pointer_to_use());
    out_value_()(coeff);
    printf("back in    > %X %X\n", in.m_data.get_cpu_p(), in.m_data.get_pointer_to_use());
    out_value_()(in);
    printf("back out   > %X %X\n", out.m_data.get_cpu_p(), out.m_data.get_pointer_to_use());
    out_value_()(out);

    std::cout << "\n\n\nTEST 2\n\n\n" << std::endl;
#endif


    boost::fusion::copy(domain.storage_pointers, actual_arg_list);

#ifdef __CUDACC__
    gridtools::setup_computation<gridtools::enumtype::Cuda>::apply( actual_arg_list, domain );
#else
    gridtools::setup_computation<gridtools::enumtype::Host>::apply( actual_arg_list, domain ); //does nothing
#endif
    
    // actual_arg_list_type* arg_list_device_ptr;
    // cudaMalloc(&arg_list_device_ptr, sizeof(actual_arg_list_type));
    cudaMemcpy(arg_list_device_ptr, &actual_arg_list , sizeof(actual_arg_list_type), cudaMemcpyHostToDevice);

#ifndef NDEBUG
    printf("\n\nFROM GPU\n\n");
#endif
    print_values<<<1,1>>>(arg_list_device_ptr);
    cudaDeviceSynchronize();
#ifndef NDEBUG
    printf("\n\nDONE WITH GPU\n\n");
#endif

    domain.finalize_computation();

    coeff.m_data.update_cpu();
    in.m_data.update_cpu();
    out.m_data.update_cpu();

    cudaFree(arg_list_device_ptr);
#ifndef NDEBUG
    printf(" > %X %X\n", coeff.m_data.get_cpu_p(), coeff.m_data.get_pointer_to_use());
    out_value_()(coeff);
    printf(" > %X %X\n", in.m_data.get_cpu_p(), in.m_data.get_pointer_to_use());
    out_value_()(in);
    printf(" > %X %X\n", out.m_data.get_cpu_p(), out.m_data.get_pointer_to_use());
    out_value_()(out);
#endif

    out_value()(&host_in);
    out_value()(&host_in);
    out_value()(&host_out);
    out_value()(&host_out);
    out_value()(&host_coeff);
    out_value()(&host_coeff);

#ifndef NDEBUG
    printf("\n\nON THE HOST\n\n");
    printf(" > %X %X\n", &coeff, coeff.m_data.get_pointer_to_use());
    out_value_()(coeff);
    printf(" > %X %X\n", &in, in.m_data.get_pointer_to_use());
    out_value_()(in);
    printf(" > %X %X\n", &out, out.m_data.get_pointer_to_use());
    out_value_()(out);
#endif

    bool failed = false;
    failed |= !the_same(in, host_in);
    failed |= !the_same(out, host_out);
    failed |= !the_same(coeff, host_coeff);

#ifndef NDEBUG
    std::cout << " *** DONE ***" << std::endl;
#endif

    return failed;
}

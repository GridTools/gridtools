/*
 * File:   test_domain.cpp
 * Author: mbianco
 *
 * Created on February 5, 2014, 4:16 PM
 *
 * Test domain features, especially the working on the GPU
 */

#define BOOST_NO_CXX11_RVALUE_REFERENCES

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <boost/current_function.hpp>
#include <boost/fusion/include/nview.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/mpl/vector.hpp>

#include <stencil-composition/stencil-composition.hpp>

using gridtools::uint_t;
using gridtools::int_t;

struct out_value {
    template <typename T>
    __host__ __device__
    void operator()(gridtools::pointer<T> x) const {
        for (uint_t i=0; i<3; ++i) {
            for (uint_t j=0; j<3; ++j) {
                for (uint_t k=0; k<3; ++k) {
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
        //printf(" > %X %X\n", &stor, stor.data().get_pointer_to_use());
        for (uint_t i=0; i<3; ++i) {
            for (uint_t j=0; j<3; ++j) {
                for (uint_t k=0; k<3; ++k) {
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
    for (uint_t i=0; i<3; ++i) {
        for (uint_t j=0; j<3; ++j) {
            for (uint_t k=0; k<3; ++k) {
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


/**
 */
bool test_domain() {

#ifdef __CUDACC__
    typedef gridtools::backend< gridtools::enumtype::Cuda,
        gridtools::enumtype::GRIDBACKEND,
        gridtools::enumtype::Naive > backend_t;
#else
    typedef gridtools::backend< gridtools::enumtype::Host,
        gridtools::enumtype::GRIDBACKEND,
        gridtools::enumtype::Naive > backend_t;
#endif
    typedef typename backend_t::storage_type<double, backend_t::storage_info<0,gridtools::layout_map<0,1,2> > >::type storage_type;

    uint_t d1 = 3;
    uint_t d2 = 3;
    uint_t d3 = 3;

    typename storage_type::storage_info_type meta_(d1,d2,d3);
    storage_type in(meta_, -1, ("in"));
    storage_type out(meta_,-7.3, ("out"));
    storage_type coeff(meta_,-3.4, ("coeff"));

    storage_type host_in(meta_,-1, ("host_in"));
    storage_type host_out(meta_,-7.3, ("host_out"));
    storage_type host_coeff(meta_,-3.4, ("host_coeff"));

    // Definition of placeholders. The order of them reflect the order the user will deal with them
    // especially the non-temporary ones, in the construction of the domain

    typedef gridtools::arg<0, storage_type > p_coeff;
    typedef gridtools::arg<1, storage_type > p_in;
    typedef gridtools::arg<2, storage_type > p_out;


    for (int_t i = 0; i < d1; ++i) {
        for (int_t j = 0; j < d2; ++j) {
            for (int_t k = 0; k < d3; ++k) {
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
    typedef boost::mpl::vector</*p_lap, p_flx, p_fly*/ p_coeff, p_in, p_out> accessor_list;
    typedef boost::mpl::vector<
        gridtools::arg<0, typename storage_type::basic_type >, 
        gridtools::arg<1, typename storage_type::basic_type >, 
        gridtools::arg<2, typename storage_type::basic_type > 
    > inner_accessor_list;

    // // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
    // // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
    gridtools::domain_type<accessor_list> domain
        (boost::fusion::make_vector(&coeff, &in, &out /*,&fly, &flx*/));

    typedef boost::mpl::vector<
        gridtools::_impl::select_storage<accessor_list, boost::mpl::na>::template apply<gridtools::static_int<0> >::type,
        gridtools::_impl::select_storage<accessor_list, boost::mpl::na>::template apply<gridtools::static_int<1> >::type,
        gridtools::_impl::select_storage<accessor_list, boost::mpl::na>::template apply<gridtools::static_int<2> >::type
    > mpl_accessor_list;

    typedef boost::mpl::vector<
        gridtools::_impl::select_storage<inner_accessor_list, boost::mpl::na>::template apply<gridtools::static_int<0> >::type,
        gridtools::_impl::select_storage<inner_accessor_list, boost::mpl::na>::template apply<gridtools::static_int<1> >::type,
        gridtools::_impl::select_storage<inner_accessor_list, boost::mpl::na>::template apply<gridtools::static_int<2> >::type
    > mpl_accessor_inner_list;

    typedef typename boost::fusion::result_of::as_vector<mpl_accessor_list>::type actual_arg_list_type;
    typedef typename boost::fusion::result_of::as_vector<mpl_accessor_inner_list>::type actual_arg_list_inner_type;

    actual_arg_list_type actual_arg_list;

    //filter the non temporary meta storage pointers among the actual ones

    // build the meta array with all the mss components
    typedef typename boost::mpl::fold<
        actual_arg_list_type
        , boost::mpl::set<>
        , boost::mpl::insert<boost::mpl::_1, gridtools::pointer
                             <boost::add_const
                              <gridtools::storage2metadata
                               <boost::mpl::_2>
                               >
                              >
                             >
        >::type actual_metadata_set_t;
    typedef gridtools::metadata_set<actual_metadata_set_t> actual_metadata_list_type;
    actual_metadata_list_type actual_metadata_list;

    typedef boost::fusion::filter_view<typename boost::fusion::result_of::as_set<actual_metadata_set_t>::type,
                                       boost::mpl::not_<gridtools::is_ptr_to_tmp<boost::mpl::_1> > > t_meta_view;

    t_meta_view  meta_view(actual_metadata_list.sequence_view());

    boost::fusion::copy(domain.m_storage_pointers, actual_arg_list);

#ifdef __CUDACC__
    gridtools::setup_computation<gridtools::enumtype::Cuda>::apply( actual_arg_list, meta_view, domain );
#else
    gridtools::setup_computation<gridtools::enumtype::Host>::apply( actual_arg_list, meta_view, domain ); //does nothing
#endif


#ifndef NDEBUG
    printf("\n\nFROM GPU\n\n");
#endif
    actual_arg_list_inner_type inner_args = boost::fusion::make_vector(coeff.get_pointer_to_use(), in.get_pointer_to_use(), out.get_pointer_to_use());
    // clang-format off
    print_values<<<1,1>>>(&inner_args);
    // clang-format on
#ifdef __CUDACC__
    cudaDeviceSynchronize();
#endif
#ifndef NDEBUG
    printf("\n\nDONE WITH GPU\n\n");
#endif
    domain.finalize_computation();

    coeff.d2h_update();
    in.d2h_update();
    out.d2h_update();

    boost::fusion::copy(domain.m_storage_pointers, actual_arg_list);

#ifdef __CUDACC__
    gridtools::setup_computation<gridtools::enumtype::Cuda>::apply( actual_arg_list, meta_view, domain );
#else
    gridtools::setup_computation<gridtools::enumtype::Host>::apply( actual_arg_list, meta_view, domain ); //does nothing
#endif

    inner_args = boost::fusion::make_vector(coeff.get_pointer_to_use(), in.get_pointer_to_use(), out.get_pointer_to_use());

#ifndef NDEBUG
    printf("\n\nFROM GPU\n\n");
#endif
    // clang-format off
    print_values<<<1,1>>>(&inner_args);
    // clang-format on
#ifdef __CUDACC__
    cudaDeviceSynchronize();
#endif
#ifndef NDEBUG
    printf("\n\nDONE WITH GPU\n\n");
#endif

    domain.finalize_computation();

    coeff.d2h_update();
    in.d2h_update();
    out.d2h_update();

    out_value()(make_pointer(*host_in.get_pointer_to_use()));
    out_value()(make_pointer(*host_in.get_pointer_to_use()));
    out_value()(make_pointer(*host_out.get_pointer_to_use()));
    out_value()(make_pointer(*host_out.get_pointer_to_use()));
    out_value()(make_pointer(*host_coeff.get_pointer_to_use()));
    out_value()(make_pointer(*host_coeff.get_pointer_to_use()));

    bool failed = false;
    failed |= !the_same(in, host_in);
    failed |= !the_same(out, host_out);
    failed |= !the_same(coeff, host_coeff);

#ifndef NDEBUG
    std::cout << " *** DONE ***" << std::endl;
#endif

    return failed;
}

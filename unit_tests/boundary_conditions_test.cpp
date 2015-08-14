#include <gridtools.hpp>
#include "common/halo_descriptor.hpp"

#ifdef __CUDACC__
#include <boundary-conditions/apply_gpu.hpp>
#else
#include <boundary-conditions/apply.hpp>
#endif

#include <boundary-conditions/zero.hpp>
#include <boundary-conditions/value.hpp>
#include <boundary-conditions/copy.hpp>

using gridtools::direction;
using gridtools::sign;
using gridtools::minus_;
using gridtools::zero_;
using gridtools::plus_;

#include "stencil-composition/backend.hpp"

#include <stdlib.h>
#include <stdio.h>

#include <boost/utility/enable_if.hpp>

using namespace gridtools;
using namespace enumtype;

#ifdef __CUDACC__
#define BACKEND backend<Cuda, Block>
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<Host , Block>
#else
#define BACKEND backend<Host, Naive >
#endif
#endif


struct bc_basic {

    // relative coordinates
    template <typename Direction, typename DataField0>
    GT_FUNCTION
    void operator()(Direction,
                    DataField0 & data_field0,
                    uint_t i, uint_t j, uint_t k) const {
        data_field0(i,j,k) = i+j+k;
    }
};

#define SET_TO_ZERO                                     \
    template <typename Direction, typename DataField0>  \
    void operator()(Direction,                          \
                    DataField0 & data_field0,           \
                    uint_t i, uint_t j, uint_t k) const {        \
        data_field0(i,j,k) = 0;                         \
    }


template <sign X>
struct is_minus {
    static const bool value = (X == minus_);
};

template <typename T, typename U>
struct is_one_of {
    static const bool value = T::value || U::value;
};

struct bc_two {

    template <typename Direction, typename DataField0>
    GT_FUNCTION
    void operator()(Direction,
                    DataField0 & data_field0,
                    uint_t i, uint_t j, uint_t k) const {
        data_field0(i,j,k) = 0;
    }

    template <sign I, sign J, sign K, typename DataField0>
    GT_FUNCTION
    void operator()(direction<I,J,K>,
                    DataField0 & data_field0,
                    uint_t i, uint_t j, uint_t k,
                    typename boost::enable_if<is_one_of<is_minus<J>, is_minus<K> > >::type *dummy = 0) const {
        data_field0(i,j,k) = (i+j+k+1);
    }

    // THE CODE ABOVE IS A REPLACEMENT OF THE FOLLOWING 4 DIFFERENT SPECIALIZATIONS
    // IT IS UGLY BUT CAN SAVE QUITE A BIT OF CODE

    // template <sign I, sign K, typename DataField0>
    // void operator()(direction<I,minus,K>,
    //                 DataField0 & data_field0,
    //                 int i, int j, int k) const {
    //     data_field0(i,j,k) = i+j+k+1;
    // }

    // template <sign J, sign K, typename DataField0>
    // void operator()(direction<minus,J,K>,
    //                 DataField0 & data_field0,
    //                 int i, int j, int k) const {
    //     data_field0(i,j,k) = i+j+k+1;
    // }

    // template <sign I, typename DataField0>
    // void operator()(direction<I,minus,minus>,
    //                 DataField0 & data_field0,
    //                 int i, int j, int k) const {
    //     data_field0(i,j,k) = i+j+k+1;
    // }

    // template <typename DataField0>
    // void operator()(direction<minus,minus,minus>,
    //                 DataField0 & data_field0,
    //                 int i, int j, int k) const {
    //     data_field0(i,j,k) = i+j+k+1;
    // }
};

struct minus_predicate {
    template <sign I, sign J, sign K>
    bool operator()(direction<I,J,K>) const {
        if (I==minus_ || J==minus_ || K == minus_)
            return false;
        return true;
    }
};

bool basic() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    typedef gridtools::BACKEND::storage_type<int_t, gridtools::layout_map<0,1,2> >::type storage_type;

    // Definition of the actual data fields that are used for input/output
    storage_type in(d1,d2,d3);
    in.allocate();
    in.initialize(-1);
    in.set_name("in");

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                in(i,j,k) = 0;
            }
        }
    }

#ifndef NDEBUG
    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }
#endif

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1,1,1,d1-2,d1);
    halos[1] = gridtools::halo_descriptor(1,1,1,d2-2,d2);
    halos[2] = gridtools::halo_descriptor(1,1,1,d3-2,d3);

#ifdef __CUDACC__
    in.clone_to_gpu();
    in.h2d_update();

    gridtools::boundary_apply_gpu<bc_basic>(halos, bc_basic()).apply(in);

    in.d2h_update();
#else
    gridtools::boundary_apply<bc_basic>(halos, bc_basic()).apply(in);
#endif

#ifndef NDEBUG
    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }
#endif

    bool result = true;

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<1; ++k) {
                if (in(i,j,k) != i+j+k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=d3-1; k<d3; ++k) {
                if (in(i,j,k) != i+j+k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<1; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != i+j+k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=d2-1; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != i+j+k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != i+j+k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=d1-1; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != i+j+k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=1; i<d1-1; ++i) {
        for (uint_t j=1; j<d2-1; ++j) {
            for (uint_t k=1; k<d3-1; ++k) {
                if (in(i,j,k) != 0) {
                    result = false;
                }
            }
        }
    }

    return result;
}

bool predicate() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    typedef gridtools::BACKEND::storage_type<int_t, gridtools::layout_map<0,1,2> >::type storage_type;

    // Definition of the actual data fields that are used for input/output
    storage_type in(d1,d2,d3);
    in.allocate();
    in.initialize(-1);
    in.set_name("in");


    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                in(i,j,k) = 0;
            }
        }
    }

#ifndef NDEBUG
    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }
#endif

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1,1,1,d1-2,d1);
    halos[1] = gridtools::halo_descriptor(1,1,1,d2-2,d2);
    halos[2] = gridtools::halo_descriptor(1,1,1,d3-2,d3);

#ifdef __CUDACC__
    in.clone_to_gpu();
    in.h2d_update();

    gridtools::boundary_apply_gpu<bc_basic, minus_predicate>(halos, bc_basic(), minus_predicate()).apply(in);

    in.d2h_update();
#else
    gridtools::boundary_apply<bc_basic, minus_predicate>(halos, bc_basic(), minus_predicate()).apply(in);
#endif

#ifndef NDEBUG
    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }
#endif

    bool result = true;

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<1; ++k) {
                if (in(i,j,k) != 0) {
#ifndef NDEBUG
                    printf("%d %d %d %d\n", i,j,k, in(i,j,k));
#endif
                    result = false;
                }
            }
        }
    }

    for (uint_t i=1; i<d1; ++i) {
        for (uint_t j=1; j<d2; ++j) {
            for (uint_t k=d3-1; k<d3; ++k) {
                if (in(i,j,k) != i+j+k) {
#ifndef NDEBUG
                    printf("%d %d %d %d\n", i,j,k, in(i,j,k));
#endif
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<1; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != 0) {
#ifndef NDEBUG
                    printf("%d %d %d %d\n", i,j,k, in(i,j,k));
#endif
                    result = false;
                }
            }
        }
    }

    for (uint_t i=1; i<d1; ++i) {
        for (uint_t j=d2-1; j<d2; ++j) {
            for (uint_t k=1; k<d3; ++k) {
                if (in(i,j,k) != i+j+k) {
#ifndef NDEBUG
                    printf("%d %d %d %d\n", i,j,k, in(i,j,k));
#endif
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != 0) {
#ifndef NDEBUG
                    printf("%d %d %d %d\n", i,j,k, in(i,j,k));
#endif
                    result = false;
                }
            }
        }
    }

    for (uint_t i=d1-1; i<d1; ++i) {
        for (uint_t j=1; j<d2; ++j) {
            for (uint_t k=1; k<d3; ++k) {
                if (in(i,j,k) != i+j+k) {
#ifndef NDEBUG
                    printf("%d %d %d %d\n", i,j,k, in(i,j,k));
#endif
                    result = false;
                }
            }
        }
    }

    for (uint_t i=1; i<d1-1; ++i) {
        for (uint_t j=1; j<d2-1; ++j) {
            for (uint_t k=1; k<d3-1; ++k) {
                if (in(i,j,k) != 0) {
#ifndef NDEBUG
                    printf("%d %d %d %d\n", i,j,k, in(i,j,k));
#endif
                    result = false;
                }
            }
        }
    }

    return result;

}

bool twosurfaces() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    typedef gridtools::BACKEND::storage_type<int_t, gridtools::layout_map<0,1,2> >::type storage_type;

    // Definition of the actual data fields that are used for input/output
    storage_type in(d1,d2,d3);
    in.allocate();
    in.initialize(-1);
    in.set_name("in");

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                in(i,j,k) = 1;
            }
        }
    }

#ifndef NDEBUG
    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }
#endif

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1,1,1,d1-2,d1);
    halos[1] = gridtools::halo_descriptor(1,1,1,d2-2,d2);
    halos[2] = gridtools::halo_descriptor(1,1,1,d3-2,d3);

#ifdef __CUDACC__
    in.clone_to_gpu();
    in.h2d_update();

    gridtools::boundary_apply_gpu<bc_two>(halos, bc_two()).apply(in);

    in.d2h_update();
#else
    gridtools::boundary_apply<bc_two>(halos, bc_two()).apply(in);
#endif

#ifndef NDEBUG
        for (uint_t i=0; i<d1; ++i) {
            for (uint_t j=0; j<d2; ++j) {
                for (uint_t k=0; k<d3; ++k) {
                    printf("%d ", in(i,j,k));
                }
                printf("\n");
            }
            printf("\n");
        }
#endif

            bool result = true;

            for (uint_t i=0; i<d1; ++i) {
                for (uint_t j=0; j<d2; ++j) {
                    for (uint_t k=0; k<1; ++k) {
                        if (in(i,j,k) != i+j+k+1) {
                            printf("A %d %d %d %d\n", i,j,k, in(i,j,k));
                            result = false;
                        }
                    }
                }
            }

            for (uint_t i=0; i<d1; ++i) {
                for (uint_t j=1; j<d2; ++j) {
                    for (uint_t k=d3-1; k<d3; ++k) {
                        if (in(i,j,k) != 0) {
#ifndef NDEBUG
                            printf("%d %d %d %d\n", i,j,k, in(i,j,k));
#endif
                            result = false;
                        }
                    }
                }
            }

            for (uint_t i=0; i<d1; ++i) {
                for (uint_t j=0; j<1; ++j) {
                    for (uint_t k=0; k<d3; ++k) {
                        if (in(i,j,k) != i+j+k+1) {
#ifndef NDEBUG
                            printf("%d %d %d %d\n", i,j,k, in(i,j,k));
#endif
                            result = false;
                        }
                    }
                }
            }

            for (uint_t i=0; i<d1; ++i) {
                for (uint_t j=d2-1; j<d2; ++j) {
                    for (uint_t k=1; k<d3; ++k) {
                        if (in(i,j,k) != 0) {
#ifndef NDEBUG
                            printf("%d %d %d %d\n", i,j,k, in(i,j,k));
#endif
                            result = false;
                        }
                    }
                }
            }

            for (uint_t i=0; i<1; ++i) {
                for (uint_t j=1; j<d2; ++j) {
                    for (uint_t k=1; k<d3; ++k) {
                        if (in(i,j,k) != 0) {
#ifndef NDEBUG
                            printf("%d %d %d %d\n", i,j,k, in(i,j,k));
#endif
                            result = false;
                        }
                    }
                }
            }

            for (uint_t i=d1-1; i<d1; ++i) {
                for (uint_t j=1; j<d2; ++j) {
                    for (uint_t k=1; k<d3; ++k) {
                        if (in(i,j,k) != 0) {
#ifndef NDEBUG
                            printf("%d %d %d %d\n", i,j,k, in(i,j,k));
#endif
                            result = false;
                        }
                    }
                }
            }

            for (uint_t i=1; i<d1-1; ++i) {
                for (uint_t j=1; j<d2-1; ++j) {
                    for (uint_t k=1; k<d3-1; ++k) {
                        if (in(i,j,k) != 1) {
#ifndef NDEBUG
                            printf("%d %d %d %d\n", i,j,k, in(i,j,k));
#endif
                            result = false;
                        }
                    }
                }
            }

            return result;

}

bool usingzero_1() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    typedef gridtools::BACKEND::storage_type<int_t, gridtools::layout_map<0,1,2> >::type storage_type;

    // Definition of the actual data fields that are used for input/output
    storage_type in(d1,d2,d3);
    in.allocate();
    in.initialize(-1);
    in.set_name("in");


    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                in(i,j,k) = -1;
            }
        }
    }

#ifndef NDEBUG
    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }
#endif

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1,1,1,d1-2,d1);
    halos[1] = gridtools::halo_descriptor(1,1,1,d2-2,d2);
    halos[2] = gridtools::halo_descriptor(1,1,1,d3-2,d3);

#ifdef __CUDACC__
    in.clone_to_gpu();
    in.h2d_update();

    gridtools::boundary_apply_gpu<gridtools::zero_boundary>(halos).apply(in);

    in.d2h_update();
#else
    gridtools::boundary_apply<gridtools::zero_boundary>(halos).apply(in);
#endif

#ifndef NDEBUG
    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }
#endif

    bool result = true;

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<1; ++k) {
                if (in(i,j,k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=d3-1; k<d3; ++k) {
                if (in(i,j,k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<1; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=d2-1; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=d1-1; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=1; i<d1-1; ++i) {
        for (uint_t j=1; j<d2-1; ++j) {
            for (uint_t k=1; k<d3-1; ++k) {
                if (in(i,j,k) != -1) {
                    result = false;
                }
            }
        }
    }

    return result;

}

bool usingzero_2() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    typedef gridtools::BACKEND::storage_type<int_t, gridtools::layout_map<0,1,2> >::type storage_type;

    // Definition of the actual data fields that are used for input/output
    storage_type in(d1,d2,d3);
    in.allocate();
    in.initialize(-1);
    in.set_name("in");
    storage_type out(d1,d2,d3);
    out.allocate();
    out.initialize(-1);
    out.set_name("out");

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                in(i,j,k) = -1;
                out(i,j,k) = -1;
            }
        }
    }

#ifndef NDEBUG
    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }
#endif

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1,1,1,d1-2,d1);
    halos[1] = gridtools::halo_descriptor(1,1,1,d2-2,d2);
    halos[2] = gridtools::halo_descriptor(1,1,1,d3-2,d3);

#ifdef __CUDACC__
    in.clone_to_gpu();
    out.clone_to_gpu();
    in.h2d_update();
    out.h2d_update();

    gridtools::boundary_apply_gpu<gridtools::zero_boundary>(halos).apply(in, out);

    in.d2h_update();
    out.d2h_update();
#else
    gridtools::boundary_apply<gridtools::zero_boundary>(halos).apply(in, out);
#endif

#ifndef NDEBUG
    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }
#endif

    bool result = true;

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<1; ++k) {
                if (in(i,j,k) != 0) {
                    result = false;
                }
                if (out(i,j,k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=d3-1; k<d3; ++k) {
                if (in(i,j,k) != 0) {
                    result = false;
                }
                if (out(i,j,k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<1; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != 0) {
                    result = false;
                }
                if (out(i,j,k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=d2-1; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != 0) {
                    result = false;
                }
                if (out(i,j,k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != 0) {
                    result = false;
                }
                if (out(i,j,k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=d1-1; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != 0) {
                    result = false;
                }
                if (out(i,j,k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=1; i<d1-1; ++i) {
        for (uint_t j=1; j<d2-1; ++j) {
            for (uint_t k=1; k<d3-1; ++k) {
                if (in(i,j,k) != -1) {
                    result = false;
                }
                if (out(i,j,k) != -1) {
                    result = false;
                }
            }
        }
    }

    return result;

}


bool usingvalue_2() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    typedef gridtools::BACKEND::storage_type<int_t, gridtools::layout_map<0,1,2> >::type storage_type;

    // Definition of the actual data fields that are used for input/output
    storage_type in(d1,d2,d3);
    in.initialize(-1);
    in.set_name("in");
    storage_type out(d1,d2,d3);
    out.allocate();
    out.initialize(-1);
    out.set_name("out");

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                in(i,j,k) = -1;
                out(i,j,k) = -1;
            }
        }
    }

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1,1,1,d1-2,d1);
    halos[1] = gridtools::halo_descriptor(1,1,1,d2-2,d2);
    halos[2] = gridtools::halo_descriptor(1,1,1,d3-2,d3);

#ifdef __CUDACC__
    in.clone_to_gpu();
    out.clone_to_gpu();
    in.h2d_update();
    out.h2d_update();

    gridtools::boundary_apply_gpu<gridtools::value_boundary<int_t> >(halos, gridtools::value_boundary<int_t>(101)).apply(in, out);

    in.d2h_update();
    out.d2h_update();
#else
    gridtools::boundary_apply<gridtools::value_boundary<int_t> >(halos, gridtools::value_boundary<int_t>(101)).apply(in, out);
#endif

    bool result = true;

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<1; ++k) {
                if (in(i,j,k) != 101) {
                    result = false;
                }
                if (out(i,j,k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=d3-1; k<d3; ++k) {
                if (in(i,j,k) != 101) {
                    result = false;
                }
                if (out(i,j,k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<1; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != 101) {
                    result = false;
                }
                if (out(i,j,k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=d2-1; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != 101) {
                    result = false;
                }
                if (out(i,j,k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != 101) {
                    result = false;
                }
                if (out(i,j,k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=d1-1; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (in(i,j,k) != 101) {
                    result = false;
                }
                if (out(i,j,k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i=1; i<d1-1; ++i) {
        for (uint_t j=1; j<d2-1; ++j) {
            for (uint_t k=1; k<d3-1; ++k) {
                if (in(i,j,k) != -1) {
                    result = false;
                }
                if (out(i,j,k) != -1) {
                    result = false;
                }
            }
        }
    }

    return result;

}

bool usingcopy_3() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    typedef gridtools::BACKEND::storage_type<int_t, gridtools::layout_map<0,1,2> >::type storage_type;

    // Definition of the actual data fields that are used for input/output
    storage_type src(d1,d2,d3);
    src.allocate();
    src.initialize(-1);
    src.set_name("src");
    storage_type one(d1,d2,d3);
    one.allocate();
    one.initialize(-1);
    one.set_name("one");
    storage_type two(d1,d2,d3);
    two.allocate();
    two.initialize(-1);
    two.set_name("two");

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                src(i,j,k) = i+k+j;
                one(i,j,k) = -1;
                two(i,j,k) = 0;
            }
        }
    }

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1,1,1,d1-2,d1);
    halos[1] = gridtools::halo_descriptor(1,1,1,d2-2,d2);
    halos[2] = gridtools::halo_descriptor(1,1,1,d3-2,d3);

#ifdef __CUDACC__
    one.clone_to_gpu();
    one.h2d_update();
    two.clone_to_gpu();
    two.h2d_update();
    src.clone_to_gpu();
    src.h2d_update();

    gridtools::boundary_apply_gpu<gridtools::copy_boundary>(halos).apply(one, two, src);

    one.d2h_update();
    two.d2h_update();
#else
    gridtools::boundary_apply<gridtools::copy_boundary>(halos).apply(one, two, src);
#endif

    bool result = true;

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<1; ++k) {
                if (one(i,j,k) != i+j+k) {
                    std::cout << "1 one " << i << ", " << j << ", " << k << ": " << one(i,j,k) << " != " << i+j+k << std::endl;
                    result = false;
                }
                if (two(i,j,k) != i+j+k) {
                    std::cout << "1 two " << i << ", " << j << ", " << k << ": " << two(i,j,k) << " != " << i+j+k << std::endl;
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=d3-1; k<d3; ++k) {
                if (one(i,j,k) != i+j+k) {
                    std::cout << "2 one " << i << ", " << j << ", " << k << ": " << one(i,j,k) << " != " << i+j+k << std::endl;
                    result = false;
                }
                if (two(i,j,k) != i+j+k) {
                    std::cout << "2 two " << i << ", " << j << ", " << k << ": " << two(i,j,k) << " != " << i+j+k << std::endl;
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<1; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (one(i,j,k) != i+j+k) {
                    std::cout << "3 one " << i << ", " << j << ", " << k << ": " << one(i,j,k) << " != " << i+j+k << std::endl;
                    result = false;
                }
                if (two(i,j,k) != i+j+k) {
                    std::cout << "3 two " << i << ", " << j << ", " << k << ": " << two(i,j,k) << " != " << i+j+k << std::endl;
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=d2-1; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (one(i,j,k) != i+j+k) {
                    std::cout << "4 one " << i << ", " << j << ", " << k << ": " << one(i,j,k) << " != " << i+j+k << std::endl;
                    result = false;
                }
                if (two(i,j,k) != i+j+k) {
                    std::cout << "4 two " << i << ", " << j << ", " << k << ": " << two(i,j,k) << " != " << i+j+k << std::endl;
                    result = false;
                }
            }
        }
    }

    for (uint_t i=0; i<1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (one(i,j,k) != i+j+k) {
                    std::cout << "5 one " << i << ", " << j << ", " << k << ": " << one(i,j,k) << " != " << i+j+k << std::endl;
                    result = false;
                }
                if (two(i,j,k) != i+j+k) {
                    std::cout << "5 two " << i << ", " << j << ", " << k << ": " << two(i,j,k) << " != " << i+j+k << std::endl;
                    result = false;
                }
            }
        }
    }

    for (uint_t i=d1-1; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                if (one(i,j,k) != i+j+k) {
                    std::cout << "6 one " << i << ", " << j << ", " << k << ": " << one(i,j,k) << " != " << i+j+k << std::endl;
                    result = false;
                }
                if (two(i,j,k) != i+j+k) {
                    std::cout << "6 two " << i << ", " << j << ", " << k << ": " << two(i,j,k) << " != " << i+j+k << std::endl;
                    result = false;
                }
            }
        }
    }

    for (uint_t i=1; i<d1-1; ++i) {
        for (uint_t j=1; j<d2-1; ++j) {
            for (uint_t k=1; k<d3-1; ++k) {
                if (one(i,j,k) != -1) {
                    std::cout << "7 one " << i << ", " << j << ", " << k << ": " << one(i,j,k) << " != " << i+j+k << std::endl;
                    result = false;
                }
                if (two(i,j,k) != 0) {
                    std::cout << "7 two " << i << ", " << j << ", " << k << ": " << two(i,j,k) << " != " << i+j+k << std::endl;
                    result = false;
                }
            }
        }
    }

    return result;
}

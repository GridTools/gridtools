#include <gridtools.hpp>
#include <common/halo_descriptor.hpp>

#ifdef CUDA_EXAMPLE
#include <boundary-conditions/apply_gpu.hpp>
#else
#include <boundary-conditions/apply.hpp>
#endif

using gridtools::direction;
using gridtools::sign;
using gridtools::minus_;
using gridtools::zero_;
using gridtools::plus_;

#include <stencil-composition/backend.hpp>

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>

using namespace gridtools;
using namespace enumtype;

#ifdef CUDA_EXAMPLE
#define BACKEND backend<Cuda, Block>
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, Block>
#else
#define BACKEND backend<Host, Naive>
#endif
#endif


template <typename T>
struct direction_bc_input {
    T value;

    GT_FUNCTION
    direction_bc_input()
        : value(1)
    {}

    GT_FUNCTION
    direction_bc_input(T v)
        : value(v)
    {}

    // relative coordinates
    template <typename Direction, typename DataField0, typename DataField1>
    GT_FUNCTION
    void operator()(Direction,
                    DataField0 & data_field0, DataField1 const & data_field1,
                    uint_t i, uint_t j, uint_t k) const {
        //std::cout << "General implementation AAA" << std::endl;
        data_field0(i,j,k) = data_field1(i,j,k) * value;
        //printf("*** value = %d\n", value);
    }

    // relative coordinates
    template <sign I, sign K, typename DataField0, typename DataField1>
    GT_FUNCTION
    void operator()(direction<I, minus_, K>,
                    DataField0 & data_field0, DataField1 const & data_field1,
                    uint_t i, uint_t j, uint_t k) const {
        // std::cout << "Implementation going A-A" << std::endl;
        data_field0(i,j,k) = 88 * value;
        //printf("*m* value = %d\n", value);
    }

    // relative coordinates
    template <sign K, typename DataField0, typename DataField1>
    GT_FUNCTION
    void operator()(direction<minus_, minus_, K>,
                    DataField0 & data_field0, DataField1 const & data_field1,
                    uint_t i, uint_t j, uint_t k) const {
        //std::cout << "Implementation going --A" << std::endl;
        data_field0(i,j,k) = 77777 * value;
        //printf("mm* value = %d\n", value);
    }

    template <typename DataField0, typename DataField1>
    GT_FUNCTION
    void operator()(direction<minus_, minus_, minus_>,
                    DataField0 & data_field0, DataField1 const & data_field1,
                    uint_t i, uint_t j, uint_t k) const {
        //std::cout << "Implementation going ---" << std::endl;
        data_field0(i,j,k) = 55555 * value;
        //printf("mmm value = %d\n", value);
    }
};



int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " dimx dimy dimz\n"
               " where args are integer sizes of the data fields" << std::endl;
        return EXIT_FAILURE;
    }

    uint_t d1 = atoi(argv[1]);
    uint_t d2 = atoi(argv[2]);
    uint_t d3 = atoi(argv[3]);

    typedef gridtools::BACKEND::storage_type<int_t, gridtools::layout_map<0,1,2> >::type storage_type;

#pragma GCC diagnostic ignored "-Wwrite-strings"
    // Definition of the actual data fields that are used for input/output
    storage_type in(d1,d2,d3);
    in.initialize(-1);
    in.set_name("in");
    storage_type out(d1,d2,d3);
    out.initialize(-7);
    out.set_name("out");
    storage_type coeff(d1,d2,d3);
    coeff.initialize(8);
    coeff.set_name("coeff");
#pragma GCC diagnostic pop

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                in(i,j,k) = 0;
                out(i,j,k) = i+j+k;
            }
        }
    }

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1,1,1,d1-2,d1);
    halos[1] = gridtools::halo_descriptor(1,1,1,d2-2,d2);
    halos[2] = gridtools::halo_descriptor(1,1,1,d3-2,d3);

#ifdef CUDA_EXAMPLE
    in.clone_to_gpu();
    out.clone_to_gpu();
    in.h2d_update();
    out.h2d_update();

    gridtools::boundary_apply_gpu<direction_bc_input<uint_t> >(halos, direction_bc_input<uint_t>(2)).apply(in, out);

    in.d2h_update();
#else
    gridtools::boundary_apply<direction_bc_input<uint_t> >(halos, direction_bc_input<uint_t>(2)).apply(in, out);
#endif

    for (uint_t i=0; i<d1; ++i) {
        for (uint_t j=0; j<d2; ++j) {
            for (uint_t k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }

    // printf("\nNow doing the same but with a stateful user struct:\n\n");

    // gridtools::boundary_apply<direction_bc_input<int> >(halos, direction_bc_input<int>(2)).apply(in, out);

    // for (int i=0; i<d1; ++i) {
    //     for (int j=0; j<d2; ++j) {
    //         for (int k=0; k<d3; ++k) {
    //             printf("%d ", in(i,j,k));
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
}

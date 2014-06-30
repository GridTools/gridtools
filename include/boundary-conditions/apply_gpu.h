#pragma once

#include "../common/defs.h"
#include "direction.h"
#include <boost/preprocessor/facilities/intercept.hpp>

namespace gridtools {

    
#define GTLOOP(zz, n, nil)                                              \
    template <typename BoundaryFunction, typename Direction,            \
              BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename DataField)> \
    __global__                                                          \
    void loop_kernel(BoundaryFunction boundary_function,                \
                     Direction direction,                               \
                     BOOST_PP_ENUM_BINARY_PARAMS(BOOST_PP_INC(n), DataField, * data_field),int starti, int startj, int startk, int nx, int ny, int nz) { \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                  \
        int j = blockIdx.y * blockDim.y + threadIdx.y;                  \
        int k = blockIdx.z * blockDim.z + threadIdx.z;                  \
        if ((i<nx) && (j<ny) && (k<nz)) {                               \
            boundary_function(direction,                                \
                              BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), *data_field), \
                              i+starti, j+startj, k+startk);            \
        }                                                               \
    }

    BOOST_PP_REPEAT(GT_MAX_ARGS, GTLOOP, _)

    /**
       Struct to apply user specified boundary condition cases on data fields.

       \tparam BoundaryFunction The user class defining the operations on the boundary. It must be copy constructible.
       \tparam HaloDescriptors  The type behaving as a read only array of halo descriptors
     */
    template <typename BoundaryFunction, typename HaloDescriptors = array<halo_descriptor, 3> >
    struct boundary_apply_gpu {
    private:
        HaloDescriptors halo_descriptors;
        BoundaryFunction const boundary_function;
        static const int ntx = 8, nty = 32, ntz = 1;
        const dim3 threads;


        public:
        boundary_apply_gpu(HaloDescriptors const& hd)
            : halo_descriptors(hd)
            , boundary_function(BoundaryFunction())
            , threads(ntx, nty, ntz)
        {}

        boundary_apply_gpu(HaloDescriptors const& hd, BoundaryFunction const & bf)
            : halo_descriptors(hd)
            , boundary_function(bf)
            , threads(ntx, nty, ntz)
        {}


#define GTAPPLY_IT(z, n, nil)                                                \
        template <typename Direction, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename DataField)> \
        void apply_it(BOOST_PP_ENUM_BINARY_PARAMS(BOOST_PP_INC(n), DataField, & data_field) ) const { \
            std::cout << Direction() << std::endl;                      \
            int nx = halo_descriptors[0].loop_high_bound_outside(Direction::I) - halo_descriptors[0].loop_low_bound_outside(Direction::I) + 1; \
            int ny = halo_descriptors[1].loop_high_bound_outside(Direction::J) - halo_descriptors[1].loop_low_bound_outside(Direction::J) + 1; \
            int nz = halo_descriptors[2].loop_high_bound_outside(Direction::K) - halo_descriptors[2].loop_low_bound_outside(Direction::K) + 1; \
            int nbx = (nx + ntx - 1) / ntx;                             \
            int nby = (ny + nty - 1) / nty;                             \
            int nbz = (nz + ntz - 1) / ntz;                             \
            dim3 blocks(nbx, nby, nbz);                                 \
            printf("nx = %d, ny = %d, nz = %d\n", nx,ny,nz);            \
            loop_kernel<<<blocks,threads>>>(boundary_function,          \
                                            Direction(),                \
                                            BOOST_PP_ENUM_BINARY_PARAMS(BOOST_PP_INC(n), data_field, .gpu_object_ptr BOOST_PP_INTERCEPT ), \
                                            halo_descriptors[0].loop_low_bound_outside(Direction::I), \
                                            halo_descriptors[1].loop_low_bound_outside(Direction::J), \
                                            halo_descriptors[2].loop_low_bound_outside(Direction::K), \
                                            nx, ny, nz);                \
        }

        BOOST_PP_REPEAT(GT_MAX_ARGS, GTAPPLY_IT, _)

#define GTAPPLY(z, n, nil)                                                \
        template <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename DataField)> \
        void apply(BOOST_PP_ENUM_BINARY_PARAMS(BOOST_PP_INC(n), DataField, & data_field) ) const { \
                                                                        \
            apply_it<direction<minus, minus, minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<minus,minus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<minus,minus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            apply_it<direction<minus, zero,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<minus, zero, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<minus, zero, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            apply_it<direction<minus, plus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<minus, plus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<minus, plus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            apply_it<direction<zero,minus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<zero,minus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<zero,minus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            apply_it<direction<zero, zero,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<zero, zero, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            apply_it<direction<zero, plus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<zero, plus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<zero, plus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            apply_it<direction<plus,minus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<plus,minus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<plus,minus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            apply_it<direction<plus, zero,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<plus, zero, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<plus, zero, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            apply_it<direction<plus, plus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<plus, plus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            apply_it<direction<plus, plus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            cudaDeviceSynchronize();                                    \
        }
    
        BOOST_PP_REPEAT(GT_MAX_ARGS, GTAPPLY, _)

    };


#undef GTLOOP
#undef GTAPPLY_IT
#undef GTAPPLY

} // namespace gridtools

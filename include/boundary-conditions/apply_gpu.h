#pragma once

#include "../common/defs.h"
#include "direction.h"

namespace gridtools {


    template <typename BoundaryFunction, typename Direction, typename DataField0, typename DataField1>
    __global__
    void loop_kernel(BoundaryFunction boundary_function, 
                     Direction direction, 
                     DataField0 * data_field0, 
                     DataField1 * data_field1, 
                     int starti, int startj, int startk, 
                     int nx, int ny, int nz) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z;

        if ((i<nx) && (j<ny) && (k<nz)) {
            // printf("nx %d\n", nx);
            // printf("ny %d\n", ny);
            // printf("nz %d\n", nz);
            // printf("value %d\n", boundary_function.value);
            boundary_function(direction, *data_field0, *data_field1, i+starti, j+startj, k+startk);
        }
    }


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


        
        template <typename Direction, typename DataField0, typename DataField1>
        void apply_it(DataField0 & data_field0, DataField1 & data_field1) const {
            std::cout << Direction() << std::endl;
            int nx = halo_descriptors[0].loop_high_bound_outside(Direction::I) - halo_descriptors[0].loop_low_bound_outside(Direction::I) + 1;
            int ny = halo_descriptors[1].loop_high_bound_outside(Direction::J) - halo_descriptors[1].loop_low_bound_outside(Direction::J) + 1;
            int nz = halo_descriptors[2].loop_high_bound_outside(Direction::K) - halo_descriptors[2].loop_low_bound_outside(Direction::K) + 1;
            int nbx = (nx + ntx - 1) / ntx;
            int nby = (ny + nty - 1) / nty;
            int nbz = (nz + ntz - 1) / ntz;
            dim3 blocks(nbx, nby, nbz);
            printf("nx = %d, ny = %d, nz = %d\n", nx,ny,nz);
            loop_kernel<<<blocks,threads>>>(boundary_function,
                                            Direction(),
                                            data_field0.gpu_object_ptr,
                                            data_field1.gpu_object_ptr,
                                            halo_descriptors[0].loop_low_bound_outside(Direction::I),
                                            halo_descriptors[1].loop_low_bound_outside(Direction::J),
                                            halo_descriptors[2].loop_low_bound_outside(Direction::K),
                                            nx, ny, nz);
        }

        template <typename DataField0, typename DataField1>
        void apply(DataField0 & data_field0, DataField1 & data_field1) const {

            


            apply_it<direction<minus, minus, minus> >(data_field0, data_field1);
            apply_it<direction<minus,minus, zero> >(data_field0, data_field1);
            apply_it<direction<minus,minus, plus> >(data_field0, data_field1);

            apply_it<direction<minus, zero,minus> >(data_field0, data_field1);
            apply_it<direction<minus, zero, zero> >(data_field0, data_field1);
            apply_it<direction<minus, zero, plus> >(data_field0, data_field1);

            apply_it<direction<minus, plus,minus> >(data_field0, data_field1);
            apply_it<direction<minus, plus, zero> >(data_field0, data_field1);
            apply_it<direction<minus, plus, plus> >(data_field0, data_field1);

            apply_it<direction<zero,minus,minus> >(data_field0, data_field1);
            apply_it<direction<zero,minus, zero> >(data_field0, data_field1);
            apply_it<direction<zero,minus, plus> >(data_field0, data_field1);

            apply_it<direction<zero, zero,minus> >(data_field0, data_field1);
            apply_it<direction<zero, zero, plus> >(data_field0, data_field1);

            apply_it<direction<zero, plus,minus> >(data_field0, data_field1);
            apply_it<direction<zero, plus, zero> >(data_field0, data_field1);
            apply_it<direction<zero, plus, plus> >(data_field0, data_field1);

            apply_it<direction<plus,minus,minus> >(data_field0, data_field1);
            apply_it<direction<plus,minus, zero> >(data_field0, data_field1);
            apply_it<direction<plus,minus, plus> >(data_field0, data_field1);

            apply_it<direction<plus, zero,minus> >(data_field0, data_field1);
            apply_it<direction<plus, zero, zero> >(data_field0, data_field1);
            apply_it<direction<plus, zero, plus> >(data_field0, data_field1);

            apply_it<direction<plus, plus,minus> >(data_field0, data_field1);
            apply_it<direction<plus, plus, zero> >(data_field0, data_field1);
            apply_it<direction<plus, plus, plus> >(data_field0, data_field1);

            cudaDeviceSynchronize();
        }

    };

} // namespace gridtools

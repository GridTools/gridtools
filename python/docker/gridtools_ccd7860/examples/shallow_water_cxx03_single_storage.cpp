#if !defined(CXX11_ENABLED)

#include "shallow_water_cxx03_single_storage.hpp"
#include <iostream>

int main(int argc, char** argv)
{

    if (argc != 6) {
        std::cout << "Usage: shallow_water_<whatever> dimx dimy dimz time target_process\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    {
    gridtools::GCL_Init(argc, argv);
//    int pid=0;
//     MPI_Comm_rank(MPI_COMM_WORLD, &pid);
//     if(pid==0)
//         cudaProfilerStart();
    int ret_val = !shallow_water::test(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
//     if(pid==0)
//         cudaProfilerStop();
    }
    GCL_Finalize();
    return 0;
}
#else
int main(int argc, char** argv){}
#endif

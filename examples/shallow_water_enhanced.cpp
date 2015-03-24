//#if defined(CXX11_ENABLED) && !defined(__GNUC__) || (defined(__clang__) || (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >=9)))

#include "shallow_water_enhanced.h"
#include <iostream>

int main(int argc, char** argv)
{

    if (argc != 5) {
        std::cout << "Usage: shallow_water_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    MPI_Init(&argc, &argv);
    gridtools::GCL_Init(argc, argv);

    return !shallow_water::test(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
}
// #else
// int main(int argc, char** argv){}
// #endif

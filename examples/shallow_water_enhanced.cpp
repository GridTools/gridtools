#include "shallow_water_enhanced.h"
#include <iostream>

int main(int argc, char** argv)
{

    if (argc != 5) {
        std::cout << "Usage: shallow_water_<whatever> dimx dimy dimz timesteps\n where args are integer sizes of the data fields and the number of timesteps performed" << std::endl;
        return 1;
    }

    gridtools::GCL_Init(argc, argv);

    return !shallow_water::test(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
}

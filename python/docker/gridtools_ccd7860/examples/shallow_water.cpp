#include "shallow_water.hpp"
#include <iostream>

int main(int argc, char** argv)
{

    if (argc != 4) {
        std::cout << "Usage: shallow_water_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }
    return !shallow_water::test(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
}

#include "vertical_advection_dycore.hpp"

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::cout << "Usage: vertical_advection_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    return !vertical_advection_dycore::test(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
}

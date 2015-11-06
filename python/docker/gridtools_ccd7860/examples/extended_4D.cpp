#include "extended_4D.hpp"
#include <iostream>

int main(int argc, char** argv)
{

#ifdef CXX11_ENABLED
    if (argc != 4) {
        std::cout << "Usage: extended_4D_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    return !assembly::test(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
#else
    assert(false);
    return -1;
#endif
}

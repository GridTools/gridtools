#define PEDANTIC_DISABLED

#include "test_assembly.h"
#include <iostream>

int main(int argc, char** argv)
{

#ifdef CXX11_ENABLED
    if (argc != 4) {
        std::cout << "Usage: assembly_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    return !intrepid::test();
#else
    assert(false);
    return -1;
#endif
}

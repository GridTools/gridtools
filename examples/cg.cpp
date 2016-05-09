#include "cg.hpp"

int main(int argc, char** argv)
{
#ifdef CXX11_ENABLED
    if (argc != 5) {
        std::cout << "Usage: interface1_<whatever> dimx dimy dimz nt\n where args are integer sizes of the data fields and number of time iterations" << std::endl;
        return 1;
    }

    return cg::solver(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));

#else
    assert(false);
    return -1;
#endif
}

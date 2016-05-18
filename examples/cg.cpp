#include "cg.hpp"

int main(int argc, char** argv)
{
#ifdef CXX11_ENABLED
    if (argc != 6) {
        std::cout << "Usage: interface1_<whatever> dimx dimy dimz maxit eps\n where args are integer sizes of the data fields, max number of iterations and eps is required tolerance" << std::endl;
        return 1;
    }

    return cg::solver(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), std::stod(argv[5]));

#else
    assert(false);
    return -1;
#endif
}

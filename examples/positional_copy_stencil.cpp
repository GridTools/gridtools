#include "positional_copy_stencil.hpp"

int main(int argc, char** argv)
{

    if (argc != 4) {
        std::cout << "Usage: copy_stencil_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    return !positional_copy_stencil::test(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
}

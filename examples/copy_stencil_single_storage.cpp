#include "copy_stencil_single_storage.hpp"

int main(int argc, char** argv)
{

    if (argc != 4) {
        std::cout << "Usage: copy_stencil_single_storage<whatever> dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    return !copy_stencil::test(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
}

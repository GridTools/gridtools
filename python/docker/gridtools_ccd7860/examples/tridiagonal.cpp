#include "tridiagonal.hpp"

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::cout << "Usage: interface1_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    return tridiagonal::solver(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
}

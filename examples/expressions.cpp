#include "expressions.h"

int_t main(int argc, char** argv)
{
#if __cplusplus>=201103L

    if (argc != 4) {
        std::cout << "Usage: expresisons_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    return !test_interface(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
#endif //#if __cplusplus>=201103L
}

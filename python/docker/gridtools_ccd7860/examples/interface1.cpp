#include "interface1.hpp"

int main(int argc, char** argv)
{

    if (argc != 4) {
        printf( "Usage: interface1_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields\n" );
        return 1;
    }

    return !horizontal_diffusion::test(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
}

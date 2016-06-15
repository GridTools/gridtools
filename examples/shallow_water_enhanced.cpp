/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "shallow_water_enhanced.hpp"
#include <iostream>

int main(int argc, char** argv)
{

    if (argc != 5) {
        std::cout << "Usage: shallow_water_<whatever> dimx dimy dimz timesteps\n where args are integer sizes of the data fields and the number of timesteps performed" << std::endl;
        return 1;
    }

    return !shallow_water::test(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
}

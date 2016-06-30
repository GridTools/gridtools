#include <iostream>
#include <common/defs.h>
#include "virtual_storage.hpp"
#include "array_addons.hpp"

using gridtools::uint_t;

int main() {
    gridtools::virtual_storage<gridtools::layout_map<0,1,2>> v_storage(gridtools::array<uint_t, 3>(34, 12, 17));

    for (int i=0; i < v_storage.dims<0>(); ++i) {
        for (int j=0; j < v_storage.dims<1>(); ++j) {
            for (int k=0; k < v_storage.dims<2>(); ++k) {
                bool result = v_storage.offset2indices(v_storage._index(i,j,k))
                    == gridtools::array<int, 3>{i,j,k};
                if (!result) {
                    std::cout << "Error: index = " << v_storage._index(i,j,k) << ", "
                              << v_storage.offset2indices(v_storage._index(i,j,k)) << " != "
                              << gridtools::array<int, 3>{i,j,k}
                    << std::endl;
                    std::cout << std::boolalpha << result << std::endl;
                }
                assert(result);
            }
        }
    }
    std::cout << "SUCCESS!" << std::endl;
    return 0;
}

#include <small_storage.h>
#include <iostream>

#define PRINT(x) std::cout << #x << " " << x << std::endl

bool test_smallstorage_indices() {

    typedef gridtools::layout_map<1,0,2> layout;

    gridtools::small_storage<int, layout, 15, 10, 5> x;

#ifndef NDEBUG
    PRINT(x._index(1,0,0));
    PRINT(x._index(0,1,0));
    PRINT(x._index(0,0,1));
#endif

    bool result = x._index(1,0,0) == 5;
    result = result && (x._index(0,1,0) == 75);
    result = result && (x._index(0,0,1) == 1);

    return result;
}


// int main() {

//     test_smallstorage_indices();

//     return 0;
// }

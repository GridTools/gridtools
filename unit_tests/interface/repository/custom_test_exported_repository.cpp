#include <gtest/gtest.h>

extern "C" void call_repository(); // implemented in test_repository.f90
TEST(repository_with_custom_getter_prefix, fortran_bindings) {
    // the test for this code is in exported_repository.cpp
    call_repository();
}

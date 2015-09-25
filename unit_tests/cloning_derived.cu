#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include "common/gpu_clone.hpp"
#include "storage/hybrid_pointer.hpp"

using namespace gridtools;

template <typename t_derived>
struct base : public clonable_to_gpu<t_derived> {
    uint_t m_size;

    base(uint_t s) : m_size(s) {}

    __host__ __device__
    base(base const& other) // default construct clonable_to_gpu
        : m_size(other.m_size)
    {  }
};

template <typename value_type>
struct derived: public base<derived<value_type> > {
    hybrid_pointer<value_type> data;

    derived(uint_t s)
        : base<derived<value_type> >(s)
        , data(s)
    {
        for (uint_t i = 0; i < data.get_size(); ++i)
            data[i] = data.get_size()-i;
        data.update_gpu();
    }

    __host__ __device__
    derived(derived const& other)
        : base<derived<value_type> >(other)
        , data(other.data)
    {  }

};

int main(int argc, char** argv) {

    if(argc < 2) {
        printf("ERROR: must pass a buffer size.\n\tUsage: %s [buffer size]\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *pend = 0;
    uint_t buffer_size = strtol(argv[1], &pend, 10);
    if(buffer_size == 0 || pend == 0 || *pend != '\0' || errno == ERANGE) {
        printf("ERROR: invalid buffer size.\n\tUsage: %s [buffer size]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int_t res = EXIT_SUCCESS;

    derived<uint_t> a(buffer_size);
    for(uint_t i = 0; i < a.data.get_size(); ++i) {
        if(a.data[i] != buffer_size - i)
            res = EXIT_FAILURE;
    }

    a.clone_to_gpu();
    a.data.update_gpu();

    for(uint_t i = 0; i < a.data.get_size(); ++i) {
        if(a.data[i] != buffer_size - i)
            res = EXIT_FAILURE;
    }

    for(uint_t i = 0; i < a.data.get_size(); ++i) {
        if(a.data[i] != buffer_size - i)
            res = EXIT_FAILURE;
    }

    cudaDeviceSynchronize();
    a.clone_from_gpu();
    a.data.update_cpu();

    for(uint_t i = 0; i < a.data.get_size(); ++i) {
        if(a.data[i] != buffer_size - i + 1)
            res = EXIT_FAILURE;
    }

    return res;
}

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include "common/gpu_clone.hpp"
#include "storage/hybrid_pointer.hpp"

using namespace gridtools;
#define _BLABLA(y) #y
#define BLABLA(z) _BLABLA(z:%d\n)
#define SIZE(x) printf( BLABLA(x) , sizeof(x) );


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

__host__ __device__
void printwhatever(derived<uint_t> * ptr) {
    printf("Ecco %X ", ptr);
    // printf("%d ", ptr->m_size);
    // printf("%d ", ptr->data.size);
    // printf("%X ", ptr->data.pointer_to_use);
    printf("%X ", ptr->data.get_cpu_p());
    printf("%X\n", ptr->data.get_gpu_p());
    for (uint_t i = 0; i < ptr->data.get_size(); ++i) {
#ifdef __CUDA_ARCH__
        ptr->data[i]++;
#endif
        printf("%d, ", ptr->data[i]);
    }
    printf("\n");
}

template <typename the_type>
__global__
void print_on_gpu(the_type * ptr) {
    printwhatever(ptr);
}

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

    SIZE(uint_t);
    SIZE(uint_t*);
    SIZE(derived<uint_t>*);
    SIZE(derived<uint_t>);
    SIZE(base<derived<uint_t> >);
    SIZE(hybrid_pointer<uint_t>);

    std::cout << "Initialize" << std::endl;
    int_t res = EXIT_SUCCESS;

    derived<uint_t> a(buffer_size);
    for(uint_t i = 0; i < a.data.get_size(); ++i) {
        if(a.data[i] != buffer_size - i)
            res = EXIT_FAILURE;
    }

    a.clone_to_gpu();
    a.data.update_gpu();

    std::cout << "Printing Beginning " << std::hex
              << a.data.get_cpu_p() << " "
              << a.data.get_gpu_p() << " "
              // << a.data.pointer_to_use << " "
              // << a.m_size << " "
              // << a.data.size << " "
              << std::dec
              << std::endl;

    printwhatever(&a);
    for(uint_t i = 0; i < a.data.get_size(); ++i) {
        if(a.data[i] != buffer_size - i)
            res = EXIT_FAILURE;
    }

    print_on_gpu<<<1,1>>>(a.gpu_object_ptr);
    for(uint_t i = 0; i < a.data.get_size(); ++i) {
        if(a.data[i] != buffer_size - i)
            res = EXIT_FAILURE;
    }

    std::cout << "Synchronize" << std::endl;
    cudaDeviceSynchronize();
    a.clone_from_gpu();
    a.data.update_cpu();

    printwhatever(&a);
    for(uint_t i = 0; i < a.data.get_size(); ++i) {
        if(a.data[i] != buffer_size - i + 1)
            res = EXIT_FAILURE;
    }

    std::cout << "Printing End " << std::hex
              << a.data.get_cpu_p() << " "
              << a.data.get_gpu_p() << " "
              // << a.data.pointer_to_use << " "
              // << a.m_size << " "
              // << a.data.size << " "
              << std::dec
              << std::endl;

    return res;
}

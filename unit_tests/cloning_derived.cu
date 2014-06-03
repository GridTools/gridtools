#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <common/gpu_clone.h>
#include <storage/hybrid_pointer.h>

using namespace gridtools;
#define _BLABLA(y) #y
#define BLABLA(z) _BLABLA(z:%d\n)
#define SIZE(x) printf( BLABLA(x) , sizeof(x) );


template <typename t_derived>
struct base : public clonable_to_gpu<t_derived> {
    int m_size;

    base(int s) : m_size(s) {}

    __host__ __device__
    base(base const& other) // default construct clonable_to_gpu
        : m_size(other.m_size)
    {  }
};

template <typename value_type>
struct derived: public base<derived<value_type> > {
    hybrid_pointer<value_type> data;

    derived(int s)
        : base<derived<value_type> >(s)
        , data(s)
    {
        for (int i = 0; i < data.size; ++i)
            data[i] = data.size-i;
        data.update_gpu();
    }

    __host__ __device__
    derived(derived const& other)
        : base<derived<value_type> >(other)
        , data(other.data)
    {  }

};

__host__ __device__
void printwhatever(derived<int> * ptr) {
    printf("Ecco %X ", ptr);
    printf("%d ", ptr->m_size);
    printf("%d ", ptr->data.size);
    printf("%X ", ptr->data.pointer_to_use);
    printf("%X ", ptr->data.cpu_p);
    printf("%X\n", ptr->data.gpu_p);
    for (int i = 0; i < ptr->data.size; ++i) {
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
    int buffer_size = strtol(argv[1], &pend, 10);
    if(buffer_size == 0 || pend == 0 || *pend != '\0' || errno == ERANGE) {
        printf("ERROR: invalid buffer size.\n\tUsage: %s [buffer size]\n", argv[0]);
        return EXIT_FAILURE;
    }

    SIZE(int);
    SIZE(int*);
    SIZE(derived<int>*);
    SIZE(derived<int>);
    SIZE(base<derived<int> >);
    SIZE(hybrid_pointer<int>);

    std::cout << "Initialize" << std::endl;
    int res = EXIT_SUCCESS;

    derived<int> a(buffer_size);
    for(int i = 0; i < a.data.size; ++i) {
        if(a.data[i] != buffer_size - i)
            res = EXIT_FAILURE;
    }

    a.clone_to_gpu();
    a.data.update_gpu();

    std::cout << "Printing Beginning " << std::hex
              << a.data.cpu_p << " "
              << a.data.gpu_p << " "
              << a.data.pointer_to_use << " "
              << a.m_size << " "
              << a.data.size << " "
              << std::dec
              << std::endl;

    printwhatever(&a);
    for(int i = 0; i < a.data.size; ++i) {
        if(a.data[i] != buffer_size - i)
            res = EXIT_FAILURE;
    }

    print_on_gpu<<<1,1>>>(a.gpu_object_ptr);
    for(int i = 0; i < a.data.size; ++i) {
        if(a.data[i] != buffer_size - i)
            res = EXIT_FAILURE;
    }

    std::cout << "Synchronize" << std::endl;
    cudaDeviceSynchronize();
    a.clone_from_gpu();
    a.data.update_cpu();

    printwhatever(&a);
    for(int i = 0; i < a.data.size; ++i) {
        if(a.data[i] != buffer_size - i + 1)
            res = EXIT_FAILURE;
    }

    std::cout << "Printing End " << std::hex
              << a.data.cpu_p << " "
              << a.data.gpu_p << " "
              << a.data.pointer_to_use << " "
              << a.m_size << " "
              << a.data.size << " "
              << std::dec
              << std::endl;

    return res;
}

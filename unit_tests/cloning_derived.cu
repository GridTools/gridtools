#include <iostream>
#include <gpu_clone.h>
#include <hybrid_pointer.h>
#include <stdio.h>

using namespace gridtools;

template <typename derived>
struct base : public clonable_to_gpu<derived> {
    int m_size;

    base(int s) : m_size(s) {}
};

template <typename value_type>
struct derived: public base<derived<value_type> > {
    hybrid_pointer<value_type> data;

    derived(int s)
        : base<derived<value_type> >(s)
        , data(s)
    {
        data[0] = 3210;
        data.update_gpu();
    }
};


//template <typename the_type>
__host__ __device__
void printwhatever(derived<int> const * ptr) {
    printf("Ecco %X %d, %X\n", ptr, ptr->m_size, (ptr->data.pointer_to_use));
}

template <typename the_type>
__global__
void print_on_gpu(the_type const* ptr) {
    printwhatever(ptr);
}


int main() {

    std::cout << "Initialize" << std::endl;

    derived<int> a(77);

    a.clone_to_gpu();
    a.data.update_gpu();
    std::cout << "Printing" << std::hex
              << a.data.cpu_p << " "
              << a.data.gpu_p << " "
              << a.data.pointer_to_use << " "
              << std::dec
              << std::endl;

    printwhatever(&a);

    print_on_gpu<<<1,1>>>(a.gpu_object_ptr);

    std::cout << "Synchronize" << std::endl;

    std::cout << "Printing" << std::hex
              << a.data.cpu_p << " "
              << a.data.gpu_p << " "
              << a.data.pointer_to_use << " "
              << std::dec
              << std::endl;

    cudaDeviceSynchronize();
    return 0;
}

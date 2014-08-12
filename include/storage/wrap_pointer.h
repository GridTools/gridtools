#pragma once

/** This class wraps a raw pointer*/
namespace gridtools {


template <typename T>
struct wrap_pointer{

    GT_FUNCTION
    wrap_pointer(wrap_pointer const& other)
        : cpu_p(other.cpu_p)
        {}

    GT_FUNCTION
    void update_gpu() {}//\todo find a way to remove this method

    GT_FUNCTION
    wrap_pointer(int size) {
        allocate_it(size);


#ifndef NDEBUG
            printf(" - %X %d\n", cpu_p, size);
#endif
        }

    GT_FUNCTION
    void allocate_it(int size){
        cpu_p = new T[size];
    }

    GT_FUNCTION
    void free_it() {
        delete cpu_p;
    }


        __host__ __device__
        operator T*() {
            return cpu_p;
        }

        __host__ __device__
        operator T const*() const {
            return cpu_p;
        }

        __host__ __device__
        T& operator[](int i) {
            assert(i>=0);
            // printf(" [%d %e] ", i, cpu_p[i]);
            return cpu_p[i];
        }

        __host__ __device__
        T const& operator[](int i) const {
            assert(i>=0);
            // printf(" [%d %e] ", i, cpu_p[i]);

            return cpu_p[i];
        }

        __host__ __device__
        T& operator*() {
            return *cpu_p;
        }

        __host__ __device__
        T const& operator*() const {
            return *cpu_p;
        }

        __host__ __device__
        T* operator+(int i) {
            return &cpu_p[i];
        }

        __host__ __device__
        T* const& operator+(int i) const {
            return &cpu_p[i];
        }

protected:
    T * cpu_p;



};
}//namespace gridtools

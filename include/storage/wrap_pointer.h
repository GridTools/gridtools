#pragma once

/** This class wraps a raw pointer*/
namespace gridtools {

    namespace workaround_ {
        template <typename T>
        struct new_op;

#define NEW_OP(x) template <>                   \
        struct new_op<x> {                      \
            x* operator()(int size) const {     \
                return new x[size];             \
            }                                   \
        };

        NEW_OP(int)
        NEW_OP(unsigned int)
        NEW_OP(char)
        NEW_OP(float)
        NEW_OP(double)
    }


template <typename T>
struct wrap_pointer{

    GT_FUNCTION
    explicit wrap_pointer(wrap_pointer const& other)
        : cpu_p(other.cpu_p)
        {}

    GT_FUNCTION
    explicit wrap_pointer(T* p)
        : cpu_p(p)
        , managed(false)
    {}

    GT_FUNCTION
    void update_gpu() {}//\todo find a way to remove this method

    GT_FUNCTION
    wrap_pointer(int size)
        : managed(true)
    {
        allocate_it(size);


#ifndef NDEBUG
            printf(" - %X %d\n", cpu_p, size);
#endif
        }

    GT_FUNCTION
    void allocate_it(int size){
#if (CUDA_VERSION > 5050)
        cpu_p = new T[size];
#else
        cpu_p = workaround_::new_op<T>()(size);
#endif
    }

    GT_FUNCTION
    void free_it() {
        if (managed) {
            delete [] cpu_p;
        }
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

    GT_FUNCTION
    const T* get_cpu_p(){return cpu_p;};

protected:
    T * cpu_p;
    bool managed;


};
}//namespace gridtools

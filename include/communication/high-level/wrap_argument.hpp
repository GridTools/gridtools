#ifndef _WRAP_ARGUMENT_H_
#define _WRAP_ARGUMENT_H_

struct wrap_argument {
    int data[27];

    __host__ __device__
    wrap_argument(int const *ptr)
    {
        for (int i=0; i<27; ++i)
            data[i] = ptr[i];
    }

    __host__ __device__
    int& operator[](int i) {
        return data[i];
    }

    __host__ __device__
    int const & operator[](int i) const {
        return data[i];
    }
};

#endif


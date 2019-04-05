/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <gridtools/common/make_array.hpp>

#define USE_DOUBLE false

template <typename T, typename lmap>
struct array {
    T *ptr;
    int n, m, l;

    array(T *_p, int _n, int _m, int _l)
        : ptr(_p), n(gridtools::make_array(_n, _m, _l)[lmap::template find<0>()]),
          m(gridtools::make_array(_n, _m, _l)[lmap::template find<1>()]),
          l(gridtools::make_array(_n, _m, _l)[lmap::template find<2>()]) {}

    T &operator()(int i, int j, int k) {
        // a[(DIM1+2*H)*(DIM2+2*H)*kk+ii*(DIM2+2*H)+jj]
        return ptr[l * m * gridtools::make_array(i, j, k)[lmap::template find<0>()] +
                   l * gridtools::make_array(i, j, k)[lmap::template find<1>()] +
                   gridtools::make_array(i, j, k)[lmap::template find<2>()]];
    }

    T const &operator()(int i, int j, int k) const {
        return ptr[l * m * gridtools::make_array(i, j, k)[lmap::template find<0>()] +
                   l * gridtools::make_array(i, j, k)[lmap::template find<1>()] +
                   gridtools::make_array(i, j, k)[lmap::template find<2>()]];
    }

    operator void *() const { return reinterpret_cast<void *>(ptr); }
    operator T *() const { return ptr; }
};

/** \file Example of use of halo_exchange pattern for regular
    grids. The comments in the code aim at highlight the process of
    instantiating and running a halo exchange pattern.
*/

inline int modulus(int __i, int __j) { return (((((__i % __j) < 0) ? (__j + __i % __j) : (__i % __j)))); }

/* Just and utility to print values
 */
template <typename array_t>
void printbuff(std::ostream &file, array_t const &a, int d1, int d2, int d3) {
    if (d1 <= 10 && d2 <= 10 && d3 <= 6) {
        file << "------------\n";
        for (int kk = 0; kk < d3; ++kk) {
            for (int jj = 0; jj < d2; ++jj) {
                file << "|";
                for (int ii = 0; ii < d1; ++ii) {
                    file << a(ii, jj, kk);
                }
                file << "|\n";
            }
            file << "\n\n";
        }
        file << "------------\n\n";
    }
}

template <bool use_double, typename VT = double>
struct triple_t;

template <typename VT>
struct triple_t</*use_double=*/false, VT> {

    typedef triple_t<false, VT> data_type;

    VT _x, _y, _z;
    GT_FUNCTION triple_t(VT a, VT b, VT c) : _x(a), _y(b), _z(c) {}
    GT_FUNCTION triple_t() : _x(-1), _y(-1), _z(-1) {}

    GT_FUNCTION triple_t(triple_t<false, VT> const &t) : _x(t._x), _y(t._y), _z(t._z) {}

    triple_t<false, VT> floor() {
        VT m = std::min(_x, std::min(_y, _z));

        return (m == -1) ? triple_t<false, VT>(m, m, m) : *this;
    }

    VT x() const { return _x; }
    VT y() const { return _y; }
    VT z() const { return _z; }
};

template <typename VT>
struct triple_t</*use_double=*/true, VT> {

    typedef double data_type;

    double value;

    GT_FUNCTION triple_t(int a, int b, int c)
        : value(static_cast<long long int>(a) * 100000000 + static_cast<long long int>(b) * 10000 +
                static_cast<long long int>(c)) {}

    GT_FUNCTION triple_t() : value(999999999999) {}

    GT_FUNCTION triple_t(triple_t<true, VT> const &t) : value(t.value) {}

    triple_t<true, VT> floor() {
        if (x() == 9999 || y() == 9999 || z() == 9999) {
            return triple_t<true, VT>();
        } else {
            return *this;
        }
    }

    int x() const {
        long long int cast = static_cast<long long int>(value);
        return static_cast<int>((cast / 100000000) % 10000);
    }

    int y() const {
        long long int cast = static_cast<long long int>(value);
        return static_cast<int>((cast / 10000) % 10000);
    }

    template <typename T>
    int y(T &file) const {
        long long int cast = static_cast<long long int>(value);
        file << "$#$@%! " << cast << " " << static_cast<int>((cast / 10000) % 10000) << std::endl;
        return static_cast<int>((cast / 10000) % 10000);
    }

    int z() const {
        long long int cast = static_cast<long long int>(value);
        return static_cast<int>((cast) % 10000);
    }
};

template <bool V, typename T>
triple_t<V, T> operator*(int a, triple_t<V, T> const &b) {
    return triple_t<V, T>(a * b.x(), a * b.y(), a * b.z());
}

template <bool V, typename T>
triple_t<V, T> operator+(int a, triple_t<V, T> const &b) {
    return triple_t<V, T>(a + b.x(), a + b.y(), a + b.z());
}

template <bool V, typename T>
triple_t<V, T> operator+(triple_t<V, T> const &a, triple_t<V, T> const &b) {
    return triple_t<V, T>(a.x() + b.x(), a.y() + b.y(), a.z() + b.z());
}

template <bool V, typename T>
std::ostream &operator<<(std::ostream &s, triple_t<V, T> const &t) {
    return s << " (" << t.x() << ", " << t.y() << ", " << t.z() << ") ";
}

template <bool V, typename T>
bool operator==(triple_t<V, T> const &a, triple_t<V, T> const &b) {
    return (a.x() == b.x() && a.y() == b.y() && a.z() == b.z());
}

template <bool V, typename T>
bool operator!=(triple_t<V, T> const &a, triple_t<V, T> const &b) {
    return !(a == b);
}

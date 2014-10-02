#include <common/array.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <stdlib.h>
#include <boost/timer/timer.hpp>

struct neighbor_offsets {
    gridtools::array<int, 3> m_offset;

    neighbor_offsets(int a, int b) {
        m_offset[0] = 1;
        m_offset[1] = a;
        m_offset[2] = b;
    }

    int offset(int node_index) const {
//        std::cout << node_index << " -> " << (node_index>>1)
//                  << " -> " << ((node_index&1)?m_offset[node_index>>1]:-m_offset[node_index>>1])
//                  << std::endl;
        return (node_index&1)?m_offset[node_index>>1]:-m_offset[node_index>>1];
    }
};

template <typename OffsetFunction>
struct structured_storage {
    std::vector<double> data;
    OffsetFunction offset_function;

    struct iterator : public std::iterator<std::random_access_iterator_tag, double> {
        typedef double value_type;

        std::vector<double>::iterator m_it;
        OffsetFunction const& f;

        iterator(std::vector<double>::iterator it, OffsetFunction const& f)
        : m_it(it)
        , f(f)
        {}

        double& operator[](int i) {
//            std::cout << "     double& operator[](int i) "
//                      << "i = " << i
//                      << " f.offset(i) " << f.offset(i)
//                      << std::endl;
            return *(m_it+f.offset(i));
        }

        double const& operator[](int i) const {
//            std::cout << "     double const& operator[](int i) const "
//                      << "i = " << i
//                      << " f.offset(i) " << f.offset(i)
//                      << std::endl;
            return *(m_it+f.offset(i));
        }

        double& operator*() {
            return *m_it;
        }

        iterator & operator++() {
            ++m_it;
            return *this;
        }

        iterator operator++(int) const {
            return m_it+1;
        }

        iterator operator+(int i) const {
            return iterator(m_it+i, f);
        }

        bool operator==(iterator const& it) const {
            return m_it==it.m_it;
        }

        bool operator!=(iterator const& it) const {
            return m_it!=it.m_it;
        }
    };

    structured_storage(int n, int m, int l)
    : data(n*m*l)
    , offset_function(m*l, l)
    { }

    iterator operator[](int i) {
        return iterator(data.begin()+i, offset_function);
    }

    iterator begin() {
        return iterator(data.begin(), offset_function);
    }

    iterator end() {
        return iterator(data.end(), offset_function);
    }

    template <typename Functor>
    double fold_neighbors(iterator it, Functor && f) const {
        double v = 0;
//        std::cout << "\nA new iteration" << std::endl;
        for (int i=0; i<6; ++i) {
//            std::cout << " ----> " << i << " : " << it[i] << std::endl;
            v = f(v, it[i]);
        }
        return v;
    }
};

int main(int argc, char** argv) {
    int n=atoi(argv[1]);
    int m=atoi(argv[2]);
    int l=atoi(argv[3]);
    std::cout << n << ", "
              << m << ", "
              << l
              << std::endl;

    structured_storage<neighbor_offsets> storage(n,m,l);
    structured_storage<neighbor_offsets> lap(n,m,l);
    structured_storage<neighbor_offsets> lap_cool(n,m,l);

    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            for (int k=0; k<l; ++k) {
                storage .data[i*m*l+j*l+k] = i*k*k+i*i*k*j-k*i;
                lap     .data[i*m*l+j*l+k] = 0;
                lap_cool.data[i*m*l+j*l+k] = 0;
            }
        }
    }

    for (int i=0; i<n; i+=std::max(1,n/5)) {
        for (int j=0; j<m; j+=std::max(1,m/5)) {
            for (int k=0; k<l; k+=std::max(1,l/10)) {
                std::cout << std::setw(5) << storage.data[i*m*l+j*l+k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    boost::timer::cpu_timer time_lap;
    for (int i=1; i<n-1; ++i) {
        for (int j=1; j<m-1; ++j) {
            for (int k=1; k<l-1; ++k) {
                lap.data[i*m*l+j*l+k] = 6*storage.data[i*m*l+j*l+k] -
                        (storage.data[(i+1)*m*l+j*l+k]+storage.data[(i-1)*m*l+j*l+k]
                        +storage.data[i*m*l+(j+1)*l+k]+storage.data[i*m*l+(j-1)*l+k]
                        +storage.data[i*m*l+j*l+k+1]+storage.data[i*m*l+j*l+k-1]);
            }
        }
    }
    boost::timer::cpu_times lapse_time_lap = time_lap.elapsed();

    std::cout << "- - - - - - - - - - - - - - - - - - - - -- -- - -- - - \n\n" << std::endl;
    for (int i=0; i<n; i+=std::max(1,n/5)) {
        for (int j=0; j<m; j+=std::max(1,m/5)) {
            for (int k=0; k<l; k+=std::max(1,l/10)) {
                std::cout << std::setw(5) << lap.data[i*m*l+j*l+k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }


    boost::timer::cpu_timer time_cool;
    for (int i=1; i<n-1; ++i) {
        for (int j=1; j<m-1; ++j) {
            for (int k=1; k<l-1; ++k) {
                lap_cool.data[i*m*l+j*l+k] = 6*storage.data[i*m*l+j*l+k] -
                        storage.fold_neighbors(storage[i*m*l+j*l+k],
                        [](double state, double value) {
//                            std::cout << "lambda " << state << " + "
//                                      << value << std::endl;
                            return state+value;
                        });
            }
        }
    }
    boost::timer::cpu_times lapse_time_cool = time_cool.elapsed();

    std::cout << "- - - - - - - - - - - - - - - - - - - - -- -- - -- - - \n\n" << std::endl;
    for (int i=0; i<n; i+=std::max(1,n/5)) {
        for (int j=0; j<m; j+=std::max(1,m/5)) {
            for (int k=0; k<l; k+=std::max(1,l/10)) {
                std::cout << std::setw(5) << lap_cool.data[i*m*l+j*l+k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    if (std::equal(lap.begin(), lap.end(), lap_cool.begin())) {
        std::cout << "PASSED!" << std::endl;
    } else {
        std::cout << "FAILED!" << std::endl;
    }

    std::cout << "TIME LAP  " << boost::timer::format(lapse_time_lap ) << std::endl;
    std::cout << "TIME COOL " << boost::timer::format(lapse_time_cool) << std::endl;
    return 0;
}

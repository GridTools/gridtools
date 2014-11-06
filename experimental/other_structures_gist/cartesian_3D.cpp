#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <stdlib.h>
#include <boost/timer/timer.hpp>

/** This is the function which defines the structure
    i.e., define the offsets of a node.
    In this case the function does not need any
    parameter since the offsets do not depend on
    the current node. The offsets depend on the
    dimensions of the grid.
*/
struct neighbor_offsets {
    int m_offset[3];
    static const int n_neighbors = 6;
    neighbor_offsets(int a, int b) {
        m_offset[0] = 1;
        m_offset[1] = a;
        m_offset[2] = b;
    }

    neighbor_offsets() {
        m_offset[0] = 0;
        m_offset[1] = 0;
        m_offset[2] = 0;
    }

    int offset(int neighbor_index) const {
        return (neighbor_index&1)?m_offset[neighbor_index>>1]:-m_offset[neighbor_index>>1];
    }
};

/** This is the storage class.
    The final plan would be to have generic
    stogare classes that will be parametrized
    with the structure function and some other
    functions to extract the needed information
    to compute the offsets, as for instance the
    'sign' bit in triangular_offsefs.
*/
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

        /** I know, using [] to access neighbors may
            seem not good, but this should highlight
            the fact that the neighbors are random-access.
        */
        double& operator[](int i) {
            return *(m_it+f.offset(i));
        }

        double const& operator[](int i) const {
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

        iterator & operator--() {
            --m_it;
            return *this;
        }

        iterator operator--(int) const {
            return m_it-1;
        }

        iterator & operator+=(int i) {
            m_it += i;
            return *this;
        }

        iterator & operator-=(int i) {
            m_it -= i;
            return *this;
        }

        iterator operator+(int i) const {
            return iterator(m_it+i, f);
        }

        iterator operator-(int i) const {
            return iterator(m_it-i, f);
        }

        bool operator==(iterator const& it) const {
            return m_it==it.m_it;
        }

        bool operator!=(iterator const& it) const {
            return m_it!=it.m_it;
        }

        bool operator<(iterator const& it) const {
            return m_it<it.m_it;
        }

        bool operator<=(iterator const& it) const {
            return m_it<=it.m_it;
        }

        bool operator>(iterator const& it) const {
            return m_it>it.m_it;
        }

        bool operator>=(iterator const& it) const {
            return m_it>=it.m_it;
        }
    };

    structured_storage(std::vector<double> && data, OffsetFunction const & offset_function)
        : data(std::move(data))
        , offset_function(offset_function)
    { }

    structured_storage()
        : data()
        , offset_function()
    { }

    iterator begin() {
        return iterator(data.begin(), offset_function);
    }

    iterator end() {
        return iterator(data.end(), offset_function);
    }

    /** This is the main function to perform operations
        (stencils) on grid elements. It takes a function to
        be applied. This function is the same in both the
        examples.
    */
    template <typename Functor>
    double fold_neighbors(iterator it, Functor && f) const {
        double v = 0;

        for (int i=0; i<OffsetFunction::n_neighbors; ++i) {
            v = f(v, it[i]);
        }
        return v;
    }
};


#ifndef INCLUDE_ONLY
int main(int argc, char** argv) {
    if (argc==1) {
        std::cout << "Usage: " << argv[0] << " n m l " << std::endl;
        std::cout << "Where n, m, l are the dimensions of the grid" << std::endl;
        return 0;
    }
    int n=atoi(argv[1]);
    int m=atoi(argv[2]);
    int l=atoi(argv[3]);

    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "                            REGULAR LAPLACIAN IN 3D" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;
    {
        std::cout << n << ", "
                  << m << ", "
                  << l
                  << std::endl;

        /** Creating the storages */
        structured_storage<neighbor_offsets> storage(std::vector<double>(n*m*l), neighbor_offsets(m*l,l));
        structured_storage<neighbor_offsets> lap(std::vector<double>(n*m*l), neighbor_offsets(m*l,l));
        structured_storage<neighbor_offsets> lap_cool(std::vector<double>(n*m*l), neighbor_offsets(m*l,l));

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

        /** **********************************************************
            These loops compute the Laplacian on the grid structure
            as if it was a C program
        */
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


        /** **********************************************************
            These loops compute the Laplacian on the grid structure
            using the fold_neighbor function.
        */
        boost::timer::cpu_timer time_cool;
        for (int i=1; i<n-1; ++i) {
            for (int j=1; j<m-1; ++j) {
                for (int k=1; k<l-1; ++k) {
                    lap_cool.data[i*m*l+j*l+k] = 6*storage.data[i*m*l+j*l+k] -
                        storage.fold_neighbors(storage.begin()+i*m*l+j*l+k,
                                               [](double state, double value) {
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
    }

    return 0;
}
#endif

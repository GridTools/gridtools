#include <cstdint>
#include <iostream>
#include <iomanip>
#include <common/layout_map.hpp>
#include <communication/halo_exchange.hpp>
#include "cuda.h"

/**
 * Struct with three properties of different type
 */
struct triplet
{
    std::uint32_t v1;
    float v2;
    double v3;

    triplet& operator=(const std::uint32_t& other)
    {
        v1 = other;
        v2 = other;
        v3 = other;
    }

    operator std::uint32_t() const
    {
        return v1;
    }

    operator float() const
    {
        return v2;
    }

    operator double() const
    {
        return v3;
    }

    bool operator==(const std::uint32_t& other) const
    {
        return (std::uint32_t(v1) == other)
            && (std::uint32_t(v2) == other)
            && (std::uint32_t(v3) == other);
    }
};

/**
 * Helper metafunction which provides the name of some types
 */
template<typename T>
struct type_name;

template<> struct type_name<float> { static constexpr const char* name = "float"; };
template<> struct type_name<double> { static constexpr const char* name = "double"; };
template<> struct type_name<triplet> { static constexpr const char* name = "triplet"; };

// Global options
static bool verbose = false;
std::ofstream out;

#ifdef NVCC
typedef gridtools::gcl_gpu arch_type;
static const bool gpu = true;
#else
typedef gridtools::gcl_cpu arch_type;
static const bool gpu = false;
#endif

/**
 * Class which contians the storage for CPU and GPU
 */
template<typename T, typename layoutmap>
class field_t
{
public:
    typedef T value_t;

    /**
     * Initialize the content of the field.
     *
     * There is a predefined content for this kind of fields. The
     * global dimension of the distributed field is assumed to be
     * 256x256x256. The value of the field at the global position
     * (i, j, k) is (i<<16) + (j<<8) + k.
     * The halos are initialized to the constant value 555555,
     * which simplifies debugging.
     *
     * \param i_size The size in I direction (not including halo)
     * \param j_size The size in J direction (not including halo)
     * \param k_size The size in K direction (not including halo)
     * \param halo Size of halo in all directions
     * \param gpu Whether to use the gpu memory or not
     * \param i_coord The coordinate of the current rank in I direction
     * \param j_coord The coordinate of the current rank in J direction
     * \param k_coord The coordinate of the current rank in K direction
     */
    field_t(int i_size, int j_size, int k_size, int halo,
            bool gpu = false,
            int i_coord = 0, int j_coord = 0, int k_coord = 0)
        : i_size_(i_size), j_size_(j_size), k_size_(k_size),
          halo_(halo), gpu_(gpu)
    {
        // Allocate memory
        int elements = (i_size+2*halo) * (j_size+2*halo) * (k_size+2*halo);
        size_ = elements*sizeof(value_t);
        data_host_ = new value_t[elements];
        if (gpu_)
        {
            cudaMalloc(&data_device_, size_);
        }

        // Compute strides (a bit of magics)
        int sizes[] = {i_size_+2*halo_, j_size_+2*halo_, k_size_+2*halo_};
        int strides[] = {0, 0, 0};

        int stride = 1;
        strides[layoutmap::template find<2>(0, 1, 2)] = stride;
        stride *= sizes[layoutmap::template find<2>(0, 1, 2)];
        strides[layoutmap::template find<1>(0, 1, 2)] = stride;
        stride *= sizes[layoutmap::template find<1>(0, 1, 2)];
        strides[layoutmap::template find<0>(0, 1, 2)] = stride;

        i_stride_ = strides[0];
        j_stride_ = strides[1];
        k_stride_ = strides[2];

        out << "\n";
        out << " - Sizes: " << i_size_ << ", " << j_size_ << ", " << k_size_ << "\n";
        out << " - Strides: " << i_stride_ << ", " << j_stride_ << ", " << k_stride_ << "\n";

        // Initialize values in the interior domain
        i_start_ = i_coord*i_size;
        j_start_ = j_coord*j_size;
        k_start_ = k_coord*k_size;

        for (int i = -halo_; i < i_size+halo_; ++i)
            for (int j = -halo_; j < j_size+halo_; ++j)
                for (int k = -halo_; k < k_size+halo_; ++k)
                {
                    data_host_[index(i, j, k)] = 555555;
                }

        for (int i = 0; i < i_size; ++i)
            for (int j = 0; j < j_size; ++j)
                for (int k = 0; k < k_size; ++k)
                {
                    const int i_global = i_start_ + i;
                    const int j_global = j_start_ + j;
                    const int k_global = k_start_ + k;

                    std::uint32_t v = (i_global<<16) + (j_global<<8) + k_global;

                    data_host_[index(i, j, k)] = v;
                }

        // Copy to device memory
        if (gpu_)
        {
            cudaMemcpy(data_device_, data_host_, size_, cudaMemcpyHostToDevice);
        }
    }

    ~field_t()
    {
        delete[] data_host_;
        if (gpu_)
            cudaFree(data_device_);
    }

    /**
     * Checks that the halo exchange has been performed correctly
     *
     * \param i_periodic Whether the field has periodic content
     *        in I direction
     * \param j_periodic Whether the field has periodic content
     *        in J direction
     * \param k_periodic Whether the field has periodic content
     *        in K direction
     *
     * \return true is returned iff the check succeeds
     */
    bool check(bool i_periodic, bool j_periodic, bool k_periodic)
    {
        if (gpu_)
        {
            cudaMemcpy(data_host_, data_device_, size_, cudaMemcpyDeviceToHost);
        }

        for (int i = -halo_; i < i_size_+halo_; ++i)
            for (int j = -halo_; j < j_size_+halo_; ++j)
                for (int k = -halo_; k < k_size_+halo_; ++k)
                {
                    int i_global = apply_periodicity(i_start_+i, i_periodic);
                    int j_global = apply_periodicity(j_start_+j, j_periodic);
                    int k_global = apply_periodicity(k_start_+k, k_periodic);

                    // Reference
                    std::uint32_t r = (i_global<<16) + (j_global<<8) + k_global;
                    if (i_global < 0 || j_global < 0 || k_global < 0)
                        r = 555555;

                    // Actual value
                    std::uint32_t v = data_host_[index(i, j, k)];

                    if (!(v == r))
                    {
                        out << " -- At (" << i_global << ", " << j_global << ", " << k_global
                            << ") found " << v << " instead of " << r << "\n";
                        return false;
                    }
                }

        return true;
    }

    /**
     * Gives access to the device data if the GPU is used
     * and to the CPU data otherwise
     *
     * \return A pointer to the beginning of the data is returned
     */
    value_t* data()
    {
        if (gpu_)
            return data_device_;
        return data_host_;
    }

    gridtools::halo_descriptor descriptor(int direction) const
    {
        int sizes[3] = {i_size_, j_size_, k_size_};
        int total[3] = {i_size_+2*halo_, j_size_+2*halo_, k_size_+2*halo_};

        return gridtools::halo_descriptor(halo_, halo_, halo_, halo_+sizes[direction]-1, total[direction]);
    }

    void print(std::ostream& out)
    {
        if (gpu_)
        {
            cudaMemcpy(data_host_, data_device_, size_, cudaMemcpyDeviceToHost);
        }

        for (int k = k_size_-1+halo_; k >= -halo_; --k)
        {
            out << k << "\n";
            for (int j = j_size_-1+halo_; j >= -halo_; --j)
            {
                out << "    ";
                for (int i = -halo_; i < i_size_+halo_; ++i)
                {
                    out << std::setw(8) << (std::uint32_t)data_host_[index(i, j, k)] << " ";
                }
                out << "\n";
            }
            out << "\n";
        }
    }

private:

    int index(int i, int j, int k) const
    {
        return (i+halo_) * i_stride_
             + (j+halo_) * j_stride_
             + (k+halo_) * k_stride_;
    }

    int apply_periodicity(int x, bool periodic)
    {
        if (periodic)
            return (x + 256) % 256;

        if (x < 0 || x >= 256)
            return -1;

        return x;
    }

    value_t *data_host_, *data_device_;
    int i_size_;
    int j_size_;
    int k_size_;
    int size_;
    int halo_;
    int i_stride_;
    int j_stride_;
    int k_stride_;
    int i_start_;
    int j_start_;
    int k_start_;
    bool gpu_;
};

template<typename T, typename layoutmap>
std::ostream& operator<<(std::ostream& out, field_t<T, layoutmap>& field)
{
    field.print(out);
    return out;
}

/**
 * Performs a halo exchange and checks the resulting data.
 *
 * \tparam layoutmap Describes the storage of data
 * \tparam use_triplet Whether to use a triplet struct (true) or
 *         a floating point type
 * \param communicator The cartesian communicator within which the
 *        exchange is done.  Must be created with the desired periodicity
 * \param halo The size of halo in each direction
 *
 * \return true is returned iff the test passed on all ranks
 */
template<typename layoutmap, typename T>
bool run(MPI_Comm communicator, int halo=1)
{
    // Get communicator information
    int dims[3], periods[3], coords[3];
    MPI_Cart_get(communicator, 3, dims, periods, coords);
    bool i_periodic = periods[0], j_periodic = periods[1], k_periodic = periods[2];
    int rank;
    MPI_Comm_rank(communicator, &rank);

    // Test name
    std::ostringstream testname;
    testname << type_name<T>::name
             << "[map=<" << layoutmap::template at<0>() << ","
                         << layoutmap::template at<1>() << ","
                         << layoutmap::template at<2>() << ">]"
            << "[periods=<" << periods[0] << ","
                            << periods[1] << ","
                            << periods[2] << ">]"
            << "[arch=" << (gpu ? "GPU" : "CPU") << "]"
            << "[halo=" << halo << "]"
        ;

    if (rank == 0)
    {
        std::cout << testname.str();
    }

    out << testname.str()
        << " on (" << coords[0] << "," << coords[1] << "," << coords[2] << ")";

    // Instantiate field
    typedef field_t<T, layoutmap> field_type;
    field_type field(256/dims[0], 256/dims[1], 256/dims[2],
                     halo, gpu, coords[0], coords[1], coords[2]
                     );

    if (verbose)
    {
        out << "Content before exchange:\n";
        out << field;
        out << "==============================================================";
        out << "\n\n";
    }

    // Define the pattern type by giving:
    //  - The data layout map
    //  - The mapping between data and grid
    //  - The data type (either floating point or structure)
    //  - The grid type
    //  - The architecture (CPU or GPU)
    //  - The version (manual packing or MPI)
    typedef gridtools::halo_exchange_dynamic_ut<
                layoutmap,
                gridtools::layout_map<0, 1, 2>,
                typename field_type::value_t,
                gridtools::MPI_3D_process_grid_t<3>,
                arch_type,
                gridtools::version_manual
            > pattern_type;

    // Instantiate halo exchange object
    pattern_type he(typename pattern_type::grid_type::period_type(
                periods[0], periods[1], periods[2]), communicator);

    // Add halo descriptors
    he.template add_halo<0>(field.descriptor(0));
    he.template add_halo<1>(field.descriptor(1));
    he.template add_halo<2>(field.descriptor(2));

    if (verbose)
    {
        out << "Descriptors:\n";
        for (int i = 0; i < 3; ++i)
        {
            gridtools::halo_descriptor hd = field.descriptor(i);
            out << " - " << hd.minus() << " " << hd.plus() << " "
                << hd.begin() << " " << hd.end() << " " << hd.total_length() << "\n";
        }
    }

    he.setup(1);

    // Perform halo exchange
    std::vector<typename field_type::value_t*> fields;
    fields.push_back(field.data());
    he.pack(fields);
    he.exchange();
    he.unpack(fields);

    if (verbose)
    {
        out << "Content after exchange:\n";
        out << field;
        out << "==============================================================";
        out << "\n\n";
    }

    // Check
    int check = field.check(periods[0], periods[1], periods[2]);
    int checkall;
    MPI_Reduce(&check, &checkall, 1, MPI_INT, MPI_LAND, 0, communicator);

    if (rank == 0)
    {
        std::cout << ": " << (checkall ? "PASSED" : "FAILED")
                  << std::endl;
    }

    out << ": " << (check ? "PASSED" : "FAILED") << std::endl;

    return checkall;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Select only first k processes, where k is the largest power of two
    // not larger than the world comm size
    int newsize = 1;
    while (newsize <= size)
        newsize <<= 1;
    newsize >>= 1;
    int do_i_test = rank < newsize;
    size = newsize;

    // Generate cartesian information
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(size, 3, dims);

    // Print some info
    if (rank == 0)
    {
        std::cout << "Running on " << (gpu ? "GPU" : "CPU")
                  << " with " << size << " processes: "
                  << dims[0] << "x" << dims[1] << "x"<< dims[2] << "\n\n";
    }

    MPI_Comm testcomm;
    MPI_Comm_split(MPI_COMM_WORLD, do_i_test, rank, &testcomm);

    // This value is only relevant on the rank 0
    bool pass = true;

    if (do_i_test)
    {
        std::ostringstream fname;
        fname << "out_" << rank << ".log";
        out.open(fname.str().c_str());

        for (int halo = 1; halo <= 4; ++halo)
        {
            for (int period = 0; period < 8; ++period)
            {
                // Create cartesian communicator
                MPI_Comm cartcomm;
                int periods[3] = {(period>>2)%2, (period>>1)%2, (period>>0)%2};
                MPI_Cart_create(testcomm, 3, dims, periods, 0, &cartcomm);

                using gridtools::layout_map;

                // Run tests with single-precision floating point
                pass = pass && run<layout_map<0, 1, 2>, float>(cartcomm, halo);
                pass = pass && run<layout_map<0, 2, 1>, float>(cartcomm, halo);
                pass = pass && run<layout_map<1, 0, 2>, float>(cartcomm, halo);
                pass = pass && run<layout_map<1, 2, 0>, float>(cartcomm, halo);
                pass = pass && run<layout_map<2, 0, 1>, float>(cartcomm, halo);
                pass = pass && run<layout_map<2, 1, 0>, float>(cartcomm, halo);

                // Run tests with double-precision floating point
                pass = pass && run<layout_map<0, 1, 2>, double>(cartcomm, halo);
                pass = pass && run<layout_map<0, 2, 1>, double>(cartcomm, halo);
                pass = pass && run<layout_map<1, 0, 2>, double>(cartcomm, halo);
                pass = pass && run<layout_map<1, 2, 0>, double>(cartcomm, halo);
                pass = pass && run<layout_map<2, 0, 1>, double>(cartcomm, halo);
                pass = pass && run<layout_map<2, 1, 0>, double>(cartcomm, halo);

                // Run tests with structure
                pass = pass && run<layout_map<0, 1, 2>, triplet>(cartcomm, halo);
                pass = pass && run<layout_map<0, 2, 1>, triplet>(cartcomm, halo);
                pass = pass && run<layout_map<1, 0, 2>, triplet>(cartcomm, halo);
                pass = pass && run<layout_map<1, 2, 0>, triplet>(cartcomm, halo);
                pass = pass && run<layout_map<2, 0, 1>, triplet>(cartcomm, halo);
                pass = pass && run<layout_map<2, 1, 0>, triplet>(cartcomm, halo);
            }
        }

        out.close();
    }

    // Finalization
    if (rank == 0)
    {
        std::cout << "\n\nOVERALL RESULT WITH " << size << " RANKS: "
                  << (pass ? "PASSED" : "FAILED") << "\n";
    }

    int ret = pass ? 0 : 1;
    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    return ret;
}


template<ushort_t D>
class mpi_cart_comm{
public:
    mpi_cart_comm():
        m_comm(),
        m_period{1},
        m_dims{0},
        m_coordinates{0}
        {}

    template<ushort_t axis>
    void set_periodicity(bool const& val){m_period[axis]=(int)val;}

    void activate_world()
        {
            int pid;
            MPI_Comm_rank(MPI_COMM_WORLD, &pid);
            int nprocs;
            MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
            MPI_Dims_create(nprocs, D, m_dims);
            MPI_Cart_create(MPI_COMM_WORLD, D, m_dims, m_period, false, &m_comm);
            MPI_Cart_get(m_comm, D, m_dims, m_period, m_coordinates);
        }

    int* coordinates(){return m_coordinates;}
    int* dimensions(){return m_dims;}
    MPI_Comm const& get(){return m_comm;}

private:
    MPI_Comm m_comm;
    int m_period[D];
    int m_dims[D];
    int m_coordinates[D];
};

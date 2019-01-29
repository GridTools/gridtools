void laplacian(storage_view_t &lap, storage_view_t &in, int boundary_size) {
    int Ni = in.total_length<0>();
    int Nj = in.total_length<1>();
    int Nk = in.total_length<2>();
    for (int i = boundary_size; i < Ni - boundary_size; ++i) {
        for (int j = boundary_size; j < Nj - boundary_size; ++j) {
            for (int k = boundary_size; k < Nk - boundary_size; ++k) {
                lap(i, j, k) = -4.0 * in(i, j, k)                  //
                               + in(i + 1, j, k) + in(i - 1, j, k) //
                               + in(i, j + 1, k) + in(i, j - 1, k);
            }
        }
    }
}

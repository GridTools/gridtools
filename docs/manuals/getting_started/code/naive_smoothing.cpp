void naive_smoothing(storage_view_t &out, storage_view_t &in, double alpha, int kmax) {
    int lap_boundary = 1;
    int full_boundary = 2;

    int Ni = in.total_length<0>();
    int Nj = in.total_length<1>();
    int Nk = in.total_length<2>();

    // Instantiate temporary fields
    storage_info_t info(Ni, Nj, Nk);
    data_store_t lap_storage(info);
    auto lap = make_host_view(lap_storage);
    data_store_t laplap_storage(info);
    auto laplap = make_host_view(laplap_storage);

    // laplacian of phi
    laplacian(lap, in, lap_boundary);
    // laplacian of lap
    laplacian(laplap, lap, full_boundary);

    for (int i = full_boundary; i < Ni - full_boundary; ++i) {
        for (int j = full_boundary; j < Nj - full_boundary; ++j) {
            for (int k = full_boundary; k < Nk - full_boundary; ++k) {
                if (k < kmax)
                    out(i, j, k) = in(i, j, k) - alpha * laplap(i, j, k);
                else
                    out(i, j, k) = in(i, j, k);
            }
        }
    }
}

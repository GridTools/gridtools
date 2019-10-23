data_store_t phi(info);
data_store_t phi_new(info);

halo_descriptor boundary_i(halo_size, halo_size, halo_size, Ni - halo_size - 1, Ni);
halo_descriptor boundary_j(halo_size, halo_size, halo_size, Nj - halo_size - 1, Nj);
auto my_grid = make_grid(boundary_i, boundary_j, axis_t{kmax, Nk - kmax});

const auto spec = [](auto phi, auto phi_new) {
    GT_DECLARE_TMP(double, lap, laplap);
    return execute_parallel()
        .stage(lap_function(), phi, lap)
        .stage(lap_function(), lap, laplap)
        .stage(smoothing_function_1(), phi, laplap, phi_new);
};

run(spec, backend_t(), my_grid, phi, phi_new);

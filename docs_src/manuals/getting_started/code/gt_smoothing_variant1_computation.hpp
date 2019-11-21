auto phi = make_storage();
auto phi_new = make_storage();
auto lap = make_storage();
auto laplap = make_storage();

halo_descriptor boundary_i(halo_size, halo_size, halo_size, Ni - halo_size - 1, Ni);
halo_descriptor boundary_j(halo_size, halo_size, halo_size, Nj - halo_size - 1, Nj);
auto my_grid = make_grid(boundary_i, boundary_j, axis_t{kmax, Nk - kmax});

const auto spec = [](auto phi, auto phi_new, auto lap, auto laplap) {
    return execute_parallel()
        .stage(lap_function(), phi, lap)
        .stage(lap_function(), lap, laplap)
        .stage(smoothing_function_1(), phi, laplap, phi_new);
};

run(spec, backend_t(), my_grid, phi, phi_new, lap, laplap);

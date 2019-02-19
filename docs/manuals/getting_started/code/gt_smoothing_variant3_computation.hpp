data_store_t phi(info);
data_store_t phi_new(info);

using arg_phi = arg<0, data_store_t>;
using arg_phi_new = arg<1, data_store_t>;
using arg_lap = tmp_arg<2, data_store_t>;

halo_descriptor boundary_i(halo_size, halo_size, halo_size, Ni - halo_size - 1, Ni);
halo_descriptor boundary_j(halo_size, halo_size, halo_size, Nj - halo_size - 1, Nj);
auto my_grid = make_grid(boundary_i, boundary_j, axis_t{kmax, Nk - kmax});

auto smoothing = make_computation<backend_t>(           //
    my_grid,                                            //
    make_multistage(                                    //
        execute::parallel(),                            //
        make_stage<lap_function>(arg_phi(), arg_lap()), //
        make_stage<smoothing_function_3>(               //
            arg_phi(),                                  //
            arg_lap(),                                  //
            arg_phi_new())                              //
        ));                                             //

smoothing.run(arg_phi{} = phi, arg_phi_new{} = phi_new);

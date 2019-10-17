data_store_t phi(info);
data_store_t phi_new(info);
data_store_t lap(info);
data_store_t laplap(info);

arg<0> arg_phi;
arg<1> arg_phi_new;
arg<2> arg_lap;
arg<3> arg_laplap;

halo_descriptor boundary_i(halo_size, halo_size, halo_size, Ni - halo_size - 1, Ni);
halo_descriptor boundary_j(halo_size, halo_size, halo_size, Nj - halo_size - 1, Nj);
auto my_grid = make_grid(boundary_i, boundary_j, axis_t{kmax, Nk - kmax});

compute<backend_t>(                                    //
    my_grid,                                           //
    arg_phi = phi,                                     //
    arg_lap = lap,                                     //
    arg_laplap = laplap,                               //
    arg_phi_new = phi_new,                             //
    make_multistage(                                   //
        execute::parallel(),                           //
        make_stage<lap_function>(arg_phi, arg_lap),    //
        make_stage<lap_function>(arg_lap, arg_laplap), //
        make_stage<smoothing_function_1>(              //
            arg_phi,                                   //
            arg_laplap,                                //
            arg_phi_new)                               //
        ));                                            //

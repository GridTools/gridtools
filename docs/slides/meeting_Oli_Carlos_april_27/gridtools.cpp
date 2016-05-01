struct geofac_div_functor {
    typedef in_accessor<1, icosahedral_topology_t::cells, extent<1> > cell_area;
    typedef in_accessor<2, icosahedral_topology_t::???, extent<1> > edge_orientation;
    typedef in_accessor<3, icosahedral_topology_t::edges, extent<1> > primal_edge_length;
    typedef inout_accessor<4, icosahedral_topology_t::edges> geofac;
    typedef boost::mpl::vector<cell_area, edge_orientation, primal_edge_length, geofac> arg_list;

    template<typename Evaluation>
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval)
    {
        auto ff = [](const double _in, const double _res) -> double { return _in + _res; };

        eval(geofac()) = eval(primal_edge_length()) * eval(edge_orientation()) / eval(cell_area());
    }
};

struct div_functor {
    typedef in_accessor<0, icosahedral_topology_t::edges, extent<1> > vn;
    typedef in_accessor<1, icosahedral_topology_t::edges, extent<1> > geofac;
    typedef inout_accessor<3, icosahedral_topology_t::cells> out_cells;
    typedef boost::mpl::vector<vn, geofac, out_cells> arg_list;

    template<typename Evaluation>
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval)
    {
        auto ff = [](const double _vn, const double _geofac, const double _res) -> double
        { return _vn * _geofac + _res; };

        eval(out_cells()) = eval(on_edges(ff, 0.0, vn(), geofac()));
    }
};

struct div_functor {
    typedef in_accessor<0, icosahedral_topology_t::edges, extent<1> > vn;
    typedef inout_accessor<1, icosahedral_topology_t::cells> out_cells;
    typedef in_accessor<2, icosahedral_topology_t::cells, extent<1> > cell_area;
    typedef in_accessor<3, icosahedral_topology_t::cells, extent<1> > edge_sign_on_cell;
    typedef in_accessor<4, icosahedral_topology_t::edges, extent<1> > primal_edge_length;
    typedef boost::mpl::vector<vn, out_cells, cell_area, edge_sign_on_cell, primal_edge_length> arg_list;

    template<typename Evaluation>
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval)
    {
        auto ff = [](const double _vn, const double _l, const double _res) -> double
        { return _vn * _l + _res; };

        eval(out_cells()) = eval(on_edges(ff, 0.0, vn(), primal_edge_length())) * eval(edge_sign_on_cell()) / eval(cell_area());
    }
};

struct div_functor {
    typedef in_accessor<0, icosahedral_topology_t::edges, extent<1> > vn;
    typedef inout_accessor<1, icosahedral_topology_t::cells> out_cells;
    typedef in_accessor<2, icosahedral_topology_t::cells, extent<1> > cell_area;
    typedef in_accessor<3, icosahedral_topology_t::edges, extent<1> > primal_edge_length;
    typedef boost::mpl::vector<vn, out_cells, cell_area, primal_edge_length> arg_list;

    template<typename Evaluation>
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval)
    {
        auto ff = [](const double _vn, const double _l, const double _res) -> double
        { return _vn * _l + _res; };

        eval(out_cells(), color=0) = eval(on_edges(ff, 0.0, vn(), primal_edge_length())) / eval(cell_area());
        eval(out_cells(), color=1) = -eval(on_edges(ff, 0.0, vn(), primal_edge_length())) / eval(cell_area());
    }
};

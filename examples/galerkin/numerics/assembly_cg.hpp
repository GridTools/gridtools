#include "assembly.hpp"

namespace gdl{
template <typename Geo>
struct assembly_cg{

public:
    using as_base_t = assembly_base<Geo>;
    using as_t = assembly<Geo>;
    using geo_map_t = typename Geo::geo_map;
    using cub_t = typename Geo::cub;
    using geometry_t = Geo;

    using grid_type = typename as_base_t::grid_type;
    using jacobian_t = typename as_t::jacobian_type;
    using storage_type = typename as_t::storage_type;
    using phi_t = typename as_t::phi_t;
    using dphi_t = typename as_t::dphi_t;
    using weights_storage_t = typename as_t::weights_storage_t;

protected:
    as_base_t m_as_base;
    as_t m_as;
    Geo m_geo;
    bool m_valid;

public:
    assembly_cg(uint_t d1_, uint_t d2_, uint_t d3_):
        m_geo(),
        m_as_base(d1_, d2_, d3_),
        m_as(m_geo, d1_, d2_, d3_),
        m_valid(true)
    {}

    void release(){
        m_valid = false;
        m_as_base.grid().release();
        m_as.jac().release();
        m_as.fe_backend().cub_weights().release();
        m_as.jac_det().release();
        m_as.jac_inv().release();
        m_as.fe_backend().val().release();
        m_as.fe_backend().grad().release();
    }

    void compute(Intrepid::EOperator op){
        assert(m_valid);
        m_geo.compute(op);
    }

    Geo& geo(){
        return m_geo;
    }

    grid_type& grid() {
        assert(m_valid);
        return m_as_base.grid();
    }

    jacobian_t& jac() {
        assert(m_valid);
        return m_as.jac();
    }

    storage_type& jac_det() {
        assert(m_valid);
        return m_as.jac_det();
    }

    jacobian_t& jac_inv() {
        assert(m_valid);
        return m_as.jac_inv();
    }

    phi_t& phi() {
        assert(m_valid);
        return m_as.fe_backend().val();
    }

    dphi_t& dphi() {
        assert(m_valid);
        return m_as.fe_backend().grad();
    }

    weights_storage_t& cub_weights() {
        assert(m_valid);
        return m_as.cub_weights();
    }
};
}//namespace gdl

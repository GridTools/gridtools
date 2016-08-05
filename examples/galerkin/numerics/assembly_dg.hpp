#include "assembly_cg.hpp"
#include "bd_assembly.hpp"

namespace gdl{

template <typename Geo>
struct assembly_dg : public assembly_cg<Geo> {
    using super = assembly_cg<Geo>;
    using geo_map_t = typename super::geo_map_t;
    using cub_t = typename super::cub_t;
    using grid_type = typename super::grid_type;

    using bd_cub_t = intrepid::boundary_cub<geo_map_t, cub_t::cubDegree>;
    using bd_discr_t = intrepid::boundary_discr<bd_cub_t>;
    using as_bd_t = bd_assembly<bd_discr_t>;
    using boundary_t = as_bd_t;

    using bd_jacobian_t = typename as_bd_t::jacobian_type;
    using face_normals_t = typename as_bd_t::face_normals_type;
    using bd_measure_t = typename as_bd_t::storage_type;
    using bd_cub_weights_t = typename as_bd_t::bd_cub_weights_t;
    using tangent_storage_t = typename as_bd_t::tangent_storage_t;
    using bd_phi_t = typename as_bd_t::phi_t;
    using bd_dphi_t = typename as_bd_t::dphi_t;

private:
    bd_cub_t m_bd_cub;
    as_bd_t m_as_bd;
    bd_discr_t m_bd_discr;

public:
    assembly_dg(uint_t d1_, uint_t d2_, uint_t d3_):
        super(d1_,d2_,d3_),
        m_bd_cub(),
        m_bd_discr(m_bd_cub,0,1,2,3,4,5),//face ordinals
        m_as_bd(m_bd_discr, d1_, d2_, d3_)
    {}

    void release(){
        super::release();
        m_as_bd.bd_jac().release();
        m_as_bd.normals().release();
        m_as_bd.bd_measure().release();
        m_as_bd.bd_backend().bd_cub_weights().release();
        m_as_bd.bd_backend().ref_normals().release();
        m_as_bd.bd_backend().val().release();
        m_as_bd.bd_backend().grad().release();
    }

    void compute(Intrepid::EOperator op){
        super::compute(op);
        m_bd_discr.compute(op, this->m_geo.get_ordering());
    }

    bd_jacobian_t& bd_jac() {
        assert(this->m_valid);
        return m_as_bd.bd_jac();
    }

    face_normals_t& normals() {
        assert(this->m_valid);
        return m_as_bd.normals();
    }

    bd_measure_t&  bd_measure() {
        assert(this->m_valid);
        return m_as_bd.bd_measure();
    }

    bd_cub_weights_t& bd_cub_weights() {
        assert(this->m_valid);
        return m_as_bd.bd_backend().bd_cub_weights();
    }

    tangent_storage_t& ref_normals() {
        assert(this->m_valid);
        return m_as_bd.bd_backend().ref_normals();
    }

    bd_phi_t& bd_phi() {
        assert(this->m_valid);
        return m_as_bd.bd_backend().val();
    }

    bd_dphi_t& bd_dphi() {
        assert(this->m_valid);
        return m_as_bd.bd_backend().grad();
    }

};
}//namespace gdl

#pragma once

#include <gridtools.h>
#include <boost/type_traits.hpp>

#include <Intrepid_Cubature.hpp>
// #include <Intrepid_FieldContainer.hpp>
#include <Intrepid_DefaultCubatureFactory.hpp>

//just some of the possible discretizations
#include "assembly.h"
#include <stencil-composition/accessor.h>
#include "intrepid_storage.h"


namespace intrepid{

    using namespace Intrepid;
    bool test(){
        // create cubature factory
        DefaultCubatureFactory<double, storage3> cubFactory;
        // set cubature degree, e.g. 2
        int cubDegree = fe::numNodes;

        // create default cubature
        Teuchos::RCP<Cubature<double, storage3> > myCub = cubFactory.create(fe::cellType, cubDegree);
        // retrieve number of cubature points
        int numCubPoints = myCub->getNumPoints();

        storage3::storage_t cub_points_s(numCubPoints, fe::spaceDim, 1);
        storage3::storage_t cub_weights_s(numCubPoints, 1, 1);
        storage3 cub_points(cub_points_s);
        storage3 cub_weights(cub_weights_s);
// intrepid_storage<storage3> cell_nodes(numCells, numNodes, spaceDim);
// intrepid_storage<storage4> jacobian(numCells, numCubPoints, spaceDim, spaceDim);
// intrepid_storage<storage4> jacobian_inv(numCells, numCubPoints, spaceDim, spaceDim);
// intrepid_storage<storage2> jacobian_det(numCells, numCubPoints);
// intrepid_storage<storage2> weighted_measure(numCells, numCubPoints);

        storage3::storage_t grad_at_cub_points_s(fe::numFields, numCubPoints, fe::spaceDim);
        storage3 grad_at_cub_points(grad_at_cub_points_s);
// intrepid_storage<storage4> transformed_grad_at_cub_points(numCells, numFields, numCubPoints, spaceDim);
// intrepid_storage<storage4> weighted_transformed_grad_at_cub_points(numCells, numFields, numCubPoints, spaceDim);
// intrepid_storage<storage3> stiffness_matrices(numCells, numFields, numFields);

        myCub->getCubature(cub_points, cub_weights);                                          // retrieve cubature points and weights
        fe::hexBasis.getValues(grad_at_cub_points, cub_points, OPERATOR_GRAD);                    // evaluate grad operator at cubature points

        assembly::test(grad_at_cub_points.get_storage(), numCubPoints );

// CellTools<double>::setJacobian(jacobian, cub_points, cell_nodes, cellType);           // compute cell Jacobians
// CellTools<double>::setJacobianInv(jacobian_inv, jacobian);                            // compute inverses of cell Jacobians
// CellTools<double>::setJacobianDet(jacobian_det, jacobian);                            // compute determinants of cell Jacobians

// FunctionSpaceTools::computeCellMeasure<double>(weighted_measure,                      // compute weighted cell measure
//                                                jacobian_det,
//                                                cub_weights);
// FunctionSpaceTools::HGRADtransformGRAD<double>(transformed_grad_at_cub_points,        // transform reference gradients into physical space
//                                                jacobian_inv,
//                                                grad_at_cub_points);
// FunctionSpaceTools::multiplyMeasure<double>(weighted_transformed_grad_at_cub_points,  // multiply with weighted measure
//                                             weighted_measure,
//                                             transformed_grad_at_cub_points);
// FunctionSpaceTools::integrate<double>(stiffness_matrices,                             // compute stiffness matrices
//                                       transformed_grad_at_cub_points,
//                                       weighted_transformed_grad_at_cub_points,
//                                       COMP_CPP);
return true;
}
}

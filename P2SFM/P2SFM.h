#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixDynamicDense;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor, int64_t> MatrixSparse;


namespace P2SFM 
{
	/*Input:
      * orig_meas: Original image coordinates (2FxN sparse matrix where missing data are [0;0])
      * options:  Structure containing options (must be initialized by ppsfm_options to contains all necessary fields)
     Output:
      * data: Matrix containing the data to compute the cost function (2Fx3N)
      * pinv_meas: Matrix containing the data for elimination of projective depths,
	    cross-product matrix or pseudo-inverse of the of the normalized homogeneous coordinates (Fx3N)
      * norm_meas: Normalized homogeneous projections coordinates (2FxN sparse matrix where missing data are [0;0])
      * visible: Visibility matrix binary mask (FxN)
      * normalisations: Normalisation transformation for each camera stacked vertically (3Fx3)
      * img_meas: Unnormaliazed homogeneous measurements of the projections, used for computing errors and scores (3FxN)*/
	void PrepareData(MatrixSparse& measurements)
	{

	}
}
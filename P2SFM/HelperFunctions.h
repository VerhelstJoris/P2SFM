#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <iostream>
#include <mat.h>;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixDynamicDense;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int64_t> MatrixSparse;

namespace Helper 
{
	//retrieve a specified variable 'varName' from a .mat file
	mxArray* ReadMxArrayFromMat(const char* filePath, const char* varName)
	{
		//TO-DO:: Fix up so return false/true

		// open MAT-file
		MATFile *pmat = matOpen(filePath, "r");
		if (pmat == NULL)
		{
			std::cout << "FILE NOT FOUND" << std::endl;
			return false;
		}

		mxArray* matrix = matGetVariable(pmat, varName);
		return matrix;
	}



	//load in a named variable from a.MAT file into an mxArray, then load this mxArray into an Eigen Dense Matrix for use later
	bool ReadDenseMatrixFromMat(const char *filePath, const char* varName,  MatrixDynamicDense* const mat)
	{
		mxArray* matrixMat = ReadMxArrayFromMat(filePath, varName);

		if (mxIsSparse(matrixMat))
		{
			std::cout << "The variable loaded in the mxArray is not a dense matrix" << std::endl;
			return false;
		}

		if (matrixMat != NULL && mxIsDouble(matrixMat) && !mxIsEmpty(matrixMat))
		{
			double *pr = mxGetPr(matrixMat); //returns pointer to data array

			const int nrCols = mxGetN(matrixMat);
			const int nrRows = mxGetNumberOfElements(matrixMat) / nrCols;

			//possible map to supplied &mat directly??
			Eigen::Map<MatrixDynamicDense> matrixEig(pr, nrRows, nrCols);
			*mat = matrixEig;

			return true;

		}

		std::cout << "Variable was not found/empty" << std::endl;
		return false;
	}

	//load in a named variable from a.MAT file into an mxArray, then load this mxArray into an Eigen Sparse Matrix for use later
	bool ReadSparseMatrixFromMat(const char* filePath, const char* varName, MatrixSparse* sparseMat)
	{
		mxArray* matrixMat = ReadMxArrayFromMat(filePath, varName);

		if (!mxIsSparse(matrixMat))
		{
			std::cout << "The variable loaded in the mxArray is not a sparse matrix" << std::endl;
			return false;
		}

		if (matrixMat != NULL && mxIsDouble(matrixMat) && !mxIsEmpty(matrixMat))
		{
			double *pr = mxGetPr(matrixMat); //returns pointer to data array
			int nrCols = mxGetN(matrixMat);
			int nrRows = mxGetNumberOfElements(matrixMat) / nrCols;
			int nnz = mxGetNzmax(matrixMat);	//max non-zero elements present in matrix

			//size_t* is the return type
			//needs to be cast to signed type for later use
			int64_t* rowIndex = reinterpret_cast<std::int64_t*>(mxGetIr(matrixMat));
			int64_t* colIndex = reinterpret_cast<std::int64_t*>(mxGetJc(matrixMat));

			Eigen::Map<MatrixSparse> spMap(nrRows, nrCols, nnz, rowIndex, colIndex, pr, 0);
			*sparseMat = spMap;
			
			return true;
		}
	}



}
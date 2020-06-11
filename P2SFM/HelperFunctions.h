#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <iostream>
#include <mat.h>;
#include "P2SFM.h"


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
	bool ReadDenseMatrixFromMat(const char *filePath, const char* varName,  MatrixDynamicDense<double>* const mat)
	{
		mxArray* matrixMat = ReadMxArrayFromMat(filePath, varName);

		if (mxIsSparse(matrixMat))
		{
			std::cout << "The variable loaded in the mxArray is not a dense matrix" << std::endl;
			return false;
		}

		if (matrixMat != NULL && mxIsDouble(matrixMat) && !mxIsEmpty(matrixMat))
		{
			double *pr = mxGetDoubles(matrixMat); //returns pointer to data array


			const int nrCols = mxGetN(matrixMat);
			const int nrRows = mxGetNumberOfElements(matrixMat) / nrCols;

			//possible map to supplied &mat directly??
			Eigen::Map<MatrixDynamicDense<double>> matrixEig(pr, nrRows, nrCols);
			*mat = matrixEig;

			return true;

		}

		std::cout << "Variable was not found/empty" << std::endl;
		return false;
	}

	//load in a named variable from a.MAT file into an mxArray, then load this mxArray into an Eigen Sparse Matrix for use later
	//TO-DO: size_t can be 32 or 64 bit depending on OS -> provide for this
	bool ReadSparseMatrixFromMat(const char* filePath, const char* varName, MatrixSparse<double>* sparseMat)
	{
		mxArray* matrixMat = ReadMxArrayFromMat(filePath, varName);

		if (!mxIsSparse(matrixMat))
		{
			std::cout << "The variable loaded in the mxArray is not a sparse matrix" << std::endl;
			return false;
		}

		if (matrixMat != NULL && mxIsDouble(matrixMat) && !mxIsEmpty(matrixMat))
		{

			double *pr = mxGetDoubles(matrixMat); //returns pointer to data array	//CORRECT

			int nrCols = mxGetN(matrixMat);			//correct
			int nrRows = mxGetNumberOfElements(matrixMat) / nrCols;	//correct
			int nnz = mxGetNzmax(matrixMat);	//max non-zero elements present in matrix	//correct

			//size_t* is the return type
			//needs to be cast to signed type for later use
			int64_t* rowIndex = reinterpret_cast<std::int64_t*>(mxGetIr(matrixMat));
			int64_t* colIndex = reinterpret_cast<std::int64_t*>(mxGetJc(matrixMat));
			
			Eigen::Map<MatrixSparse<double>> spMap(nrRows, nrCols, nnz, colIndex, rowIndex, pr,0);

			sparseMat->reserve(nnz);
			*sparseMat = spMap;

			return true;
		}

		return false;
	}
}
// P2SFM.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>


#include "HelperFunctions.h"

/*Input:
	* measurements : Original image coordinates(2FxN sparse matrix where missing data are[0; 0])
	* image_size : Size of the images, used to compute the pyramidal visibility scores(2xF)
	* centers : Principal points coordinates for each camera(2xF)
	* options : Structure containing options(must be initialized by ppsfm_options to contains all necessary fields)
*/
int main()
{

	MatrixDynamicDense *centerInput = new MatrixDynamicDense();
	MatrixDynamicDense *sizeInput = new MatrixDynamicDense();
	//const char* fileName = "Dataset/dino319.mat";
	const char* fileName = "Dataset/cherubColmap.mat";

	Helper::ReadDenseMatrixFromMat(fileName,"centers", centerInput);
	Helper::ReadDenseMatrixFromMat(fileName,"image_size", sizeInput);

	std::cout << "First 2 dense Matrices loaded in" << std::endl;
	std::cout << *centerInput << std::endl << std::endl;
	std::cout << *sizeInput << std::endl << std::endl;

	//read in sparse matrix
	MatrixSparse* measureSparse= new MatrixSparse();
	Helper::ReadSparseMatrixFromMat(fileName, "measurements", measureSparse);

	std::cout << "Sparse matrix loaded in" << std::endl;
	//std::cout << *measureSparse << std::endl;

	//std::cout << mat1;

	std::cin.get();
}



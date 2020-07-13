// P2SFM.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "HelperFunctions.h"
#include "P2SFM.h"

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

void Timer(std::chrono::time_point<std::chrono::steady_clock> t1, std::chrono::time_point<std::chrono::steady_clock> t2, const char* name)
{
	std::cout << std::endl << name << " time: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
		<< " milliseconds" << std::endl;
}



/*Input:
	* measurements : Original image coordinates(2FxN sparse matrix where missing data are[0; 0])
	* image_size : Size of the images, used to compute the pyramidal visibility scores(2xF)
	* centers : Principal points coordinates for each camera(2xF)
	* options : Structure containing options(must be initialized by ppsfm_options to contains all necessary fields)
*/
int main()
{
	MatrixDynamicDense<double> *centerInput = new MatrixDynamicDense<double>();
	MatrixDynamicDense<double> *sizeInput = new MatrixDynamicDense<double>();


	P2SFM::Options opt = P2SFM::Options();
	const char* fileName = "Dataset/dino319.mat";
	//const char* fileName = "Dataset/cherubColmap.mat";
	opt.min_common_init = 50;


	Helper::ReadDenseMatrixFromMat(fileName,"centers", centerInput);
	Helper::ReadDenseMatrixFromMat(fileName,"image_size", sizeInput);

	std::cout << "First 2 dense Matrices loaded in" << std::endl;

	//read in sparse matrix
	MatrixColSparse<double,int64_t>* measureSparse = new MatrixColSparse<double,int64_t>();
	auto result = Helper::ReadSparseMatrixFromMat(fileName, "measurements", measureSparse);

	std::cout << "Sparse matrix loaded in" << std::endl;

	
	//std::cout << "TEST MATRIX" << std::endl;
	//std::cout << testMat << std::endl;

	auto t1 = Clock::now();

	//START OF ACTUAL OPERATIONS
	//prepare data
	//Eigen::SparseMatrix<double, Eigen::ColMajor, int64_t> data, pinv_meas, norm_meas, normalisations;
	MatrixColSparse<double,int64_t> data, pinv_meas, norm_meas, normalisations;
	MatrixDynamicDense<bool> visibility;

	//PREPARE DATA
	//======================================================
	P2SFM::PrepareData(*measureSparse,data, pinv_meas,norm_meas,normalisations,visibility, opt);

	auto t2 = Clock::now();
	Timer(t1, t2, "PrepareData");

	MatrixColSparse<double, int64_t>* testMatFile = new MatrixColSparse<double, int64_t>();
	result = Helper::ReadSparseMatrixFromMat("Dataset/test.mat", "norm", testMatFile);
	
	//std::cout << "NORM FILE" << std::endl << (*testMatFile).row(0);
	//auto equality = testMatFile->isApprox(norm_meas);
	//std::cout << "equal: " << equality << std::endl;

	//PAIRS AFFINITY
	//======================================================
	std::vector<int> affinity;
	MatrixDynamicDense<int> view_pairs;
	std::tie(affinity,view_pairs) = P2SFM::PairsAffinity(*measureSparse, visibility, (*sizeInput).cast<int>(), opt);

	auto t3 = Clock::now();
	Timer(t2, t3, "PairsAffinity");

	for (size_t i = 0; i < opt.max_models; i++)
	{
		//INITIALISATION
		//=====================================================
		MatrixDynamicDense<int>estimated (0, 0);
		P2SFM::Initialisation(norm_meas, visibility, affinity, view_pairs, estimated, opt);
	}

	delete measureSparse;
	delete centerInput;
	delete sizeInput;

	std::cin.get();
}


void CreateTestMat()
{
	//TEST SPARSE MATRIX
	std::vector<Eigen::Triplet<double>> tripletList;
	const int amountElem = 90;
	//const int amountElem = 30;
	tripletList.reserve(amountElem);
	MatrixDynamicDense<bool> visibilityTest(amountElem / 30, 10);
	visibilityTest.setZero();

	for (size_t i = 0; i < amountElem; i++)
	{
		if (i % 2 == 0)
		{
			if ((i / 10 + 1) % 3 == 0)
			{
				tripletList.push_back(Eigen::Triplet<double>((int)(i / 10), i % 10, 1));
				visibilityTest.coeffRef((int)(i / 10) / 3, i % 10) = 1;
			}
			else
			{
				tripletList.push_back(Eigen::Triplet<double>((int)(i / 10), i % 10, i + 1));
			}
		}
	}
	std::cout << "TripletList created" << std::endl;

	MatrixColSparse<double, int64_t> testMat(amountElem / 10, 10);

	testMat.setFromTriplets(tripletList.begin(), tripletList.end());
	tripletList.clear();

}


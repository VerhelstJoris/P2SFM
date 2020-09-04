// P2SFM.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "HelperFunctions.h"
#include "P2SFM.h"

#include <thread>
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
	//const char* fileName = "Dataset/house.mat";
	//const char* fileName = "Dataset/cherubColmap.mat";
	opt.min_common_init = 50;
	opt.debuginform = P2SFM::Options::InformDebug::NONE;

	Helper::ReadDenseMatrixFromMat(fileName,"centers", centerInput);
	Helper::ReadDenseMatrixFromMat(fileName,"image_size", sizeInput);

	std::cout << "First 2 dense Matrices loaded in" << std::endl;

	//read in sparse matrix
	MatrixColSparse<double,int64_t>* measureSparse = new MatrixColSparse<double,int64_t>();
	auto result = Helper::ReadSparseMatrixFromMat(fileName, "measurements", measureSparse);

	std::cout << "Sparse matrix loaded in" << std::endl;

	auto t1 = Clock::now();

	auto results = P2SFM::Main(*measureSparse, *centerInput, *sizeInput, opt);

	Timer(t1, Clock::now(), " END OF PROGRAM TIME");
	
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


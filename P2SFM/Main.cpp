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
	opt.debuginform = P2SFM::Options::InformDebug::REGULAR;

	Helper::ReadDenseMatrixFromMat(fileName,"centers", centerInput);
	Helper::ReadDenseMatrixFromMat(fileName,"image_size", sizeInput);

	std::cout << "First 2 dense Matrices loaded in" << std::endl;


	//read in sparse matrix
	MatrixColSparse<double,int64_t>* measureSparse = new MatrixColSparse<double,int64_t>();
	auto result = Helper::ReadSparseMatrixFromMat(fileName, "measurements", measureSparse);

	std::cout << "Sparse matrix loaded in" << std::endl;

	auto t1 = Clock::now();

	//START OF ACTUAL OPERATIONS
	//prepare data
	//Eigen::SparseMatrix<double, Eigen::ColMajor, int64_t> data, pinv_meas, norm_meas, normalisations;
	MatrixColSparse<double,int64_t> data, pinv_meas, norm_meas, normalisations, hom_measurements;
	MatrixDynamicDense<bool> visibility;

	//PREPARE DATA
	//======================================================
	std::tie(hom_measurements,data,pinv_meas,norm_meas,normalisations,visibility) = P2SFM::PrepareData(*measureSparse, opt);

	auto t2 = Clock::now();
	Timer(t1, t2, "PrepareData");


	//PAIRS AFFINITY
	//======================================================
	std::vector<int> affinity;
	MatrixDynamicDense<int> view_pairs;
	std::tie(affinity,view_pairs) = P2SFM::PairsAffinity(hom_measurements, visibility, (*sizeInput).cast<int>(), opt);

	auto t3 = Clock::now();
    Timer(t2, t3, "PairsAffinity");

	std::cout << "VIEW PAIRS" << std::endl << view_pairs << std::endl;

	std::vector<std::tuple<P2SFM::ViewsPointsProjections<double, int64_t>,
		P2SFM::ViewsPointsProjections<double, int64_t>, MatrixColSparse<double, int64_t> >> models;



	Vector<int> estimated_views(0);
	int iteration = 0;

	for (size_t i = 0; i < opt.max_models; i++)
	{
		iteration++;

		//INITIALISATION
		//=====================================================	
		bool success;
		//use same indextype as other sparse matrices to prevent weird conversion errors
		P2SFM::ViewsPointsProjections<double, int64_t> projections_initial, projections_final;
	
		//std::cout << "ESTIMATED " << std::endl << estimated_views << std::endl;

		std::tie(projections_initial, success) = P2SFM::Initialisation(norm_meas, visibility, affinity, view_pairs, estimated_views, opt);
		if (!success)
		{
			std::cout << "failed to robsutly solve initial problem" << std::endl;
			break;	
		}

		auto t4 = Clock::now();
		Timer(t3, t4, "Initialisation");
	
		std::cout << projections_initial.pathway << std::endl;
		//std::cout << projections_initial.cameras << std::endl;
		//std::cout << projections_initial.points << std::endl; //some values in poitns have their sign flipped

		//COMPLETION
		//=====================================================
		MatrixColSparse<double, int64_t> inliers;
		std::tie(projections_final, inliers) = P2SFM::Complete(data, pinv_meas, visibility, normalisations, hom_measurements, *sizeInput, *centerInput, projections_initial, opt);
		auto t5 = Clock::now();
		Timer(t4, t5, "Completion");
	
		if (opt.debuginform >= P2SFM::Options::InformDebug::REGULAR)
		{
			std::cout << " Reconstructed";
		}
	
		//FINAL REFINEMENT
		//====================================================
		if (opt.final_refinement)
		{
			std::cout << "FINAL REFINEMENT 0-" << projections_final.pathway.size() << std::endl;
			projections_final = P2SFM::Refinement(data, pinv_meas, inliers, projections_final, 0, projections_final.pathway.size(), false, P2SFM::EigenHelpers::REFINEMENT_TYPE::FINAL, "Final", opt);
			if (opt.debuginform >= P2SFM::Options::InformDebug::REGULAR)
			{
				Timer(t1, Clock::now(), " Final Refinement");
			}
		}
	
		models.push_back({ projections_initial, projections_final, inliers });

		std::cout << "initial: " << std::endl << projections_initial.pathway << std::endl << "final: " << std::endl << projections_final.pathway << std::endl;

		std::cout << "BEFORE ESTIMATED VIEWS: " << estimated_views << std::endl;

		//append negative pathway values to estimated views + sort + remove duplicate elements
		Vector<int> cams = std::get<1>(P2SFM::EigenHelpers::GetPointsCamsPathway(projections_final.pathway, 0, projections_final.pathway.size()-1));

		//create vector containing both, including duplicates
		Vector<int> duplicates(estimated_views.size() + cams.size());
		duplicates << estimated_views, cams;
		std::sort(duplicates.data(), duplicates.data() + duplicates.size());
		estimated_views.resize(duplicates.size()); // resize and discard previous data


		int count = 0; int prevElem=-1;
		for (size_t i = 0; i < duplicates.size(); i++)
		{
			if (prevElem != duplicates(i))
			{
				estimated_views(count) = duplicates(i);
				count++;
			}

			prevElem = duplicates(i);
		}

		estimated_views.conservativeResize(count);

		std::cout << "ESTIMATED VIEWS: " << estimated_views << std::endl;


		if (estimated_views.size() == visibility.rows())
		{
			std::cout << "enough views have been estimated" << std::endl;
			break;
		}
		else if (opt.debuginform >= P2SFM::Options::InformDebug::REGULAR)
		{
			std::cout << "Incomplete reconstruction, estimated a total of " << estimated_views.size() << "/" << visibility.rows()
				<< "(" << (double)estimated_views.size() / visibility.rows() * 100.0 << "%) views over " << i+1 << " model(s)" << std::endl;
		}
	}

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


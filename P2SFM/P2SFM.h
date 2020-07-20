#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <numeric>

//dynamic sized dense matrix template 
template <typename Type> using MatrixDynamicDense = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

//dynamic sized sparse matrix template
template <typename ScalarType, typename IndexType> using MatrixColSparse = Eigen::SparseMatrix<ScalarType, Eigen::ColMajor, IndexType>;

//vector Template
template <typename Type> using Vector = Eigen::Matrix<Type,1, Eigen::Dynamic>;
template <typename Type> using VectorVertical = Eigen::Matrix<Type, Eigen::Dynamic,1>;
//array template
template <typename Type> using Array = Eigen::Array<Type, 1, Eigen::Dynamic>;


//TEST
//TODO: REMOVE
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;


namespace P2SFM 
{
	namespace EigenHelpers
	{
		//a vector is sorted in descending order, the matrix cols are then are arranged in the same manner as the vector was sorted
		template <typename VecType, typename MatrixType>
		void sortVectorAndMatrix(std::vector<VecType>&vec, Eigen::Matrix<MatrixType,Eigen::Dynamic,Eigen::Dynamic>& mat, bool rowSort=true) 
		{
			// initialize original index locations
			std::vector<size_t> idxVec(vec.size());
			std::iota(idxVec.begin(), idxVec.end(), 0);

			std::sort(idxVec.begin(), idxVec.end(),
				[&vec](size_t id1, size_t id2) {return vec[id1] > vec[id2]; });	//get vector of how indices will look like sorted

			std::sort(vec.rbegin(), vec.rend());	//actually sort the vector

			//copy temp cols into mat cols
			//TODO:
			//can prevent making a copy of the matrix by swapping cols instead and keeping track of where the real one goes
			MatrixDynamicDense<MatrixType> temp(mat);
			for (size_t i = 0; i < (rowSort?mat.rows():mat.cols()); i++)
			{
				if (rowSort)
				{
					mat.row(i).swap(temp.row(idxVec[i]));
				}
				else
				{
					mat.col(i).swap(temp.col(idxVec[i]));
				}
			}
		}

		bool PointComparison(const Eigen::Vector2i& p1, const Eigen::Vector2i& p2) {
			if (p1[0] < p2[0]) { return true; }
			if (p1[0] > p2[0]) { return false; }
			return (p1[1] < p2[1]);
		}

		bool PointEquality(const Eigen::Vector2i& p1, const Eigen::Vector2i& p2) {
			return ((p1[0] == p2[0]) && (p1[1] == p2[1]));
		}

		template <typename T>
		Vector<T> StdVecToEigenVec(std::vector<T> vec)
		{
			T* ptr = &vec[0];
			Eigen::Map<Vector<T>> map(ptr, vec.size());
			return (Vector<T>)map;
		}

		template <typename T>
		Array<T> StdVecToEigenArray(std::vector<T> vec)
		{
			T* ptr = &vec[0];
			Eigen::Map<Array<T>> map(ptr, vec.size());
			return (Array<T>)map;
		}

		enum ESTIMATION_METHOD
		{
			DEFAULT,
			RANSAC
		};

		template <typename MatrixType, typename IndexType>
		std::tuple<MatrixDynamicDense<MatrixType>, MatrixDynamicDense<MatrixType>>
		GetViewsCommonPoints(const MatrixColSparse<MatrixType, IndexType>& transposedMat, 
			const  MatrixDynamicDense<bool>& visibility, const int view1, const int view2)
		{
			auto visiblePoints = visibility.row(view1 / 3) && visibility.row(view2 / 3);

			MatrixDynamicDense<MatrixType> firstViewDense(2, visiblePoints.count());
			MatrixDynamicDense<MatrixType> secondViewDense(2, visiblePoints.count());

			int colOffset = 0;
			for (size_t i = view1; i < view1 + 2; i++)
			{
				colOffset = 0;

				//k will give data from a single column, construct submatrix from first 3 cols
				for (typename MatrixColSparse<MatrixType, IndexType>::InnerIterator firstIt(transposedMat, i); firstIt; ++firstIt)
				{
					//check if corresponding value in 'commonpoints' is true/false
					if (visiblePoints(firstIt.row()) == 1)
					{
						firstViewDense(firstIt.col() % 3, colOffset) = firstIt.value();
						colOffset++;
					}
				}
			}

			for (size_t j = view2; j < view2 + 2; j++)
			{
				colOffset = 0;

				//k will give data from a single column, construct submatrix from first 3 cols
				for (typename MatrixColSparse<MatrixType, IndexType>::InnerIterator secondIt(transposedMat, j); secondIt; ++secondIt)
				{
					//check if corresponding value in 'commonpoints' is true/false
					if (visiblePoints(secondIt.row()) == 1)
					{
						secondViewDense(secondIt.col() % 3, colOffset) = secondIt.value();
						colOffset++;
					}
				}

			}

			return { firstViewDense, secondViewDense };
		}


		//get a homogenous coordinate located at 'col' with the first row being startRow
		template <typename MatrixType, typename IndexType>
		Eigen::Matrix<MatrixType, 3, 1> GetMeasurementFromView(const MatrixColSparse<MatrixType, IndexType>& mat, const int startRow, const int col)
		{
			Eigen::Matrix<MatrixType, 3, 1> retVec;

			int count = 0;
			for (typename MatrixColSparse<MatrixType, IndexType>::InnerIterator it(mat, col); it; ++it)	//jump to correct col
			{
				if (it.row() / 3 == startRow)	//wanted row?
				{
					retVec(it.row() % 3) = it.value();
					count++;
					if (count == 3)
						break;
				}
			}

			return retVec;
		}
	}

	namespace AlgebraHelpers
	{
		/*Find a transformation to normalize some data points,
		  made of a translation so that the origin is the centroid,
		  and a rescaling so that the average distance to it is sqrt(2).
		  Input:
		    * measMat: the data points, if all the entries of every 3rd row are ones, it is assumed it is homogeneous.
			* normalizedMat: empty matrix passed by ref
			* transformMat: empty matrix passed by ref
		    * isotropic: boolean indicating if the scaling should be isotropic(default) or not.
		  Output :
		    * normalizedMat: the transformation applied to measMat.
			* transformMat: the transformation matrix itself
		*/
		template <typename MatrixType, typename IndexType>
		void NormTrans(const MatrixColSparse<MatrixType, IndexType>& measMat,
			MatrixColSparse<MatrixType, IndexType>& normalizedMat,
			MatrixColSparse<MatrixType, IndexType>& transformMat,
			bool isotropic = true)
		{
			IndexType dimensions = measMat.rows();
			normalizedMat.resize(dimensions, measMat.cols());
			normalizedMat.reserve(measMat.nonZeros());

			transformMat.reserve((3 * dimensions) / 9 * 5);
			transformMat.resize(dimensions, dimensions);


			//create array 
			Array<double> tempArr(dimensions); //temporary to store differences/centroids
			Array<double> pointsPerDimension(dimensions);
			//each view has its own scale
			Array<double> scaleArr(dimensions);

			tempArr.setZero();
			pointsPerDimension.setZero();
			scaleArr.setZero();

			bool homogeneous =true;

			//check if every value in the last row in measMat is ==1
			//cannot access a single row as we are in a colMajor format
			for (int k = 0; k < measMat.outerSize(); ++k)
			{
				for (typename MatrixColSparse<MatrixType,IndexType>::InnerIterator it(measMat, k); it; ++it)
				{
					tempArr[it.row()] += it.value();
					pointsPerDimension[it.row()]++;

					//every value in a 3rd row should have it's value be ==1 for the matrix to be homogenous already
					if ((it.row() + 1) % 3 == 0 && it.value() != (MatrixType)1)
						homogeneous = false;
						//homogenous[it.row() / 3] = false;
					
				}
			}

			////element-wise division of tempArr/pointsPerDimension to calculate mean value per row
			////every third element should have a value of 1 of the entire submat is homogenous
			tempArr /= pointsPerDimension;
			std::vector<Eigen::Triplet<MatrixType,IndexType>> tripletList;
		
			tripletList.reserve((3 * dimensions) / 9 * 5);	//exact amount of elements

			if (isotropic)
			{
				//SUM OF COL (matching x/y value) ,leave out third col if homogenous
				//scale = sqrt(2) . / mean(sqrt(sum(diff. ^ 2, 1)));
				// mean every row(sqrt(sum per row(element wise power for every value in diff)))
				for (int k = 0; k < measMat.outerSize(); ++k)
				{
					for (typename MatrixColSparse<MatrixType,IndexType>::InnerIterator it(measMat, k); it; ++it)
					{
						auto prevVal = it.value();
						++it;
						scaleArr[it.row()/3] += (sqrt(pow(it.value() - tempArr[it.row()],2) + pow(prevVal - tempArr[it.row() - 1],2)))/pointsPerDimension[it.row()] ;
						++it;	//skip over third row
					}
				}

				scaleArr = sqrt(2) / scaleArr;

				//create matrix containing transformations for a single view (3 rows)
				// scale	   0		-centroid.x *  scale
				//	 0		scale		-centroid.y *  scale
				//	 0		   0					1
				//then apply this matrix to the relevant view for a final 'normalized matrix'
				for (size_t i = 0; i < scaleArr.size(); i += 3)	//rework indexing
				{
					tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(i, i, scaleArr[i/3])); //x-scale
					tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(i+1, i + 1, scaleArr[i/3])); //y-scale
					tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(i+2, i + 2, (MatrixType)1)); //1
					tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(i, i + 2, -tempArr[i] * scaleArr[i/3]));	   //-centroid.x *  scale
					tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(i + 1, i + 2, -tempArr[i + 1] * scaleArr[i / 3]));  //-centroid.y *  scale
				}


				transformMat.setFromTriplets(tripletList.begin(), tripletList.end());
			}
			else
			{

				for (int k = 0; k < measMat.outerSize(); ++k)
				{
					for (typename MatrixColSparse<MatrixType,IndexType>::InnerIterator it(measMat, k); it; ++it)
					{
						scaleArr[it.row()] += abs(it.value() - tempArr[it.row()]);	//can this be moved to the earlier iteration??
						//problem being the fact it needs to be the absolute value
					}
				}
		

				//calculate the mean of the abs values, per ROW
				scaleArr = sqrt(2) / (scaleArr/pointsPerDimension);

				//create matrix containing transformations for a single view (3 rows)
				// x-scale	   0		-centroid.x * x scale
				//	 0		y-scale		-centroid.y * y scale
				//	 0		   0					1
				//then apply this matrix to the relevant view for a final 'normalized matrix'
				for (size_t i = 0; i < scaleArr.size() ; i+=3)	//rework indexing
				{
					tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(i, i, scaleArr[i])); //x-scale
					tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(i+1, i+1, scaleArr[i + 1])); //y-scale
					tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(i+2, i+2, (MatrixType)1)); //1
					tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(i, i+2, -tempArr[i] * scaleArr[i])); //-centroid.x * x scale
					tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(i+1, i+2, -tempArr[i + 1] * scaleArr[i+1])); //-centroid.y * y scale
				}

				transformMat.setFromTriplets(tripletList.begin(), tripletList.end());
			}

			normalizedMat = transformMat * measMat;
			if (!homogeneous)
			{
				//all values in 3rd rows to 1
				for (int k = 0; k < normalizedMat.outerSize(); ++k)
				{
					for (typename MatrixColSparse<MatrixType,IndexType>::InnerIterator it(normalizedMat, k); it; ++it)
					{
						if ((it.row() + 1) % 3 == 0)
							it.valueRef() = (MatrixType)1;
					}
				}
			}
		}


		//construct a matrix
		//TO-DO:
		//ADD A MORE INDEPTH DESCRIPTION
		template <typename MatrixType>
		inline Eigen::Matrix<MatrixType, 3, 3> CrossProductMatrix(const Eigen::Matrix<MatrixType, 3, 1>& measEntry)
		{
			//M = [0 - v(3) v(2); v(3) 0 - v(1); -v(2) v(1) 0];
			Eigen::Matrix<MatrixType, 3, 3> M;
			M << 0, -measEntry.coeff(2, 0), -measEntry.coeff(1, 0),
				measEntry.coeff(2, 0), 0, -measEntry.coeff(0, 0),
				-measEntry.coeff(1, 0), measEntry.coeff(0, 0), 0;
			return M;
		}

		//TO-DO:
		//ADD A MORE INDEPTH DESCRIPTION
		template <typename MatrixType>
		MatrixDynamicDense<MatrixType> Estimate(const MatrixDynamicDense<MatrixType>& coeffs, bool enforceRank=true)
		{
			//singular value decomposition
			//BDCSVD implementation is preferred for larger matrices (>16)
			//ComputeFullV because the very last column is the one wanted
			Eigen::BDCSVD<MatrixDynamicDense<MatrixType>> svd(coeffs, Eigen::ComputeFullV);

			//get last col -svd.matrixV() and put in 3*3 matrix
			MatrixDynamicDense<MatrixType> fundamental_matrix(3, 3);
			auto lastCol = svd.matrixV().col(svd.matrixV().cols() - 1);
			fundamental_matrix << lastCol(0), lastCol(3), lastCol(6),
				lastCol(1), lastCol(4), lastCol(7),
				lastCol(2), lastCol(5), lastCol(8);

			//is this still needed?
			//if (coeffs.rows() <= coeffs.cols())
			//{
			//	//m <= n — svd(A,0) is equivalent to svd(A) MATLAB
			//	auto lastCol = svd.matrixV().col(svd.matrixV().cols() - 1);
			//	fundamental_matrix << lastCol(0), lastCol(3), lastCol(6),
			//		lastCol(1), lastCol(4), lastCol(7),
			//		lastCol(2), lastCol(5), lastCol(8);
			//}
			//else
			//{
			//	//m > n — svd(A, 0) is equivalent to svd(A, 'econ').
			//	//[U,S,V] = svd(A,'econ') produces an economy-size decomposition of m-by-n matrix A:
			//	//m > n — Only the first n columns of U are computed, and S is n - by - n.
			//	//m = n — svd(A, 'econ') is equivalent to svd(A).
			//	//m < n — Only the first m columns of V are computed, and S is m - by - m.
			//
			//	//work with the m>n case as amount of rows should be a lot more than amount of cols
			//	auto lastCol = svd.matrixV().col(coeffs.rows());
			//	fundamental_matrix << lastCol(0), lastCol(3), lastCol(6),
			//		lastCol(1), lastCol(4), lastCol(7),
			//		lastCol(2), lastCol(5), lastCol(8);
			//	
			//}

			if (enforceRank)
			{
				//JacobiSVD is preferable for smaller matrices
				//thin or full should have no effect as we're working with a 3x3 mat
				Eigen::JacobiSVD<MatrixDynamicDense<MatrixType>> svd2(fundamental_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
			
				MatrixDynamicDense<MatrixType> singularMat(3, 3);	//plug singular values in matrix for use in calculations
				singularMat << svd2.singularValues()[0], 0, 0,
					0, svd2.singularValues()[1], 0,
					0, 0, 0;
			
				//col 1 and 3 wrong sign for these 2 but this does not matter in next calculation
				MatrixDynamicDense<MatrixType> matU = svd2.matrixU();
				MatrixDynamicDense<MatrixType> matV = svd2.matrixV();
			
				matV.transposeInPlace();
			
				fundamental_matrix = matU * singularMat * matV;
			}

			return fundamental_matrix;
		}

		//TO-DO:
		//ADD A MORE INDEPTH DESCRIPTION
		template <typename MatrixType>
		std::tuple<VectorVertical<MatrixType>, MatrixDynamicDense<MatrixType>>
			RansacEstimate(const MatrixDynamicDense<MatrixType>& coeffs,
			const double confidence,
			const int max_iterations,
			const double distance_threshold)
		{
			const int num_projs = coeffs.rows();
			int best_score = INT_MAX;
			VectorVertical<MatrixType> best_inliers(num_projs);
			MatrixDynamicDense<MatrixType> best_estim(3, 3);

			int last_iteration = max_iterations;	//use in for-loop -> can be changed

			double log_conf = log(1 - confidence / 100);

			//std::vector<int> test_set(8,-1); //initialize to -1

			for (size_t i = 0; i <= last_iteration; i++)
			{
				//create random permutation of 8 values between 0 - (num_projs-1) without repeating elements
				//std::generate(test_set.begin(), test_set.end(), [num_projs,&test_set]() 
				//{ 
				//	int num;
				//	do
				//	{
				//		num = rand() % (num_projs - 1);
				//	} while (std::find(test_set.begin(),test_set.end(),num)!=test_set.end());
				//	
				//	return num; 
				//});

				std::vector<int> test_set{ 12, 89, 93, 102, 124, 136, 106, 79 };	//TEST SET

				//create 'submatrix' coeffs with only the test set rows
				MatrixDynamicDense<MatrixType> subMat(8, coeffs.cols());


				for (size_t j = 0; j < test_set.size(); j++)
				{
					subMat.row(j) = coeffs.row(test_set[j]);
				}


				auto estimation = Estimate(subMat, false);

				//estimation to col vector order [0 3 6 1 4 7 2 5 8]
				VectorVertical<MatrixType> estimateVec(9);
				estimateVec << estimation(0, 0), estimation(1, 0), estimation(2, 0),
					estimation(0, 1), estimation(1, 1), estimation(2, 1),
					estimation(0, 2), estimation(1, 2), estimation(2, 2);

				if ((subMat * estimateVec).norm() / sqrt(8) < distance_threshold)	//THIS MIGHT NOT BE CORRECT YET
				{
					//move down into loop
					VectorVertical<MatrixType> errors = (coeffs * estimateVec).array().pow(2);
					VectorVertical<MatrixType> inliers(errors.rows());
					inliers.setZero();
					inliers = (errors.array() < distance_threshold).select(errors, inliers);

					//inliers.nonzeros() does not work as expected with dense matrices
					int nnzCount = (inliers.array() != MatrixType(0) ).count();
					double score = inliers.array().sum() + (inliers.rows() - nnzCount) * distance_threshold;

					//std::cout << "SCORE: " << score << " NNZ_COUNT: " << nnzCount << std::endl;

					//is this the best match yet? (with enough elements present)
					if (nnzCount > 8 && score < 7 * best_score)
					{
						//estimate all inliers coeffs	
						//reuse subMat for this
						subMat.resize(nnzCount, coeffs.cols());
						int count = 0;
						for (size_t j = 0; j < inliers.size(); j++)	//get all rows from coeffs that are inliers
						{
							if(inliers[j] != 0)
							{
								subMat.row(count) = coeffs.row(j);	
								count++;
							}
						}

						//can reuse old estimation var as we know the output matrix is 3x3
						estimation = Estimate(subMat, false);
						estimateVec << estimation(0, 0), estimation(1, 0), estimation(2, 0),
							estimation(0, 1), estimation(1, 1), estimation(2, 1),
							estimation(0, 2), estimation(1, 2), estimation(2, 2);

						errors.resize(subMat.rows(), 1);
						errors = (subMat * estimateVec).array().pow(2);

						score = errors.array().sum() + (inliers.rows() - nnzCount)*distance_threshold;

						//the lower the score the better, save best for return
						if (score < best_score)
						{
							best_score = score;
							best_estim = estimation;
							best_inliers = inliers;

							//adaptive maximum numbers of iterations
							double ratio = (double)nnzCount / num_projs;
							const double minVal = std::pow(2, -52);	//min difference between 2 values in matlab
							double prob = std::max(minVal, std::min(1 - minVal, 1 - std::pow(ratio,8))); //clamp the ratio value


							last_iteration = std::min(int(std::ceil(log_conf / log(prob))), max_iterations);
						}
					}

				}
			}


			//best inliers has at least options.min_common_init amount of projection correspondences

			return { best_inliers ,best_estim };
		}


		/*
		Compute Vectorization of fundamental matrix (a' * F * b = 0)
		Input: 
			*a: a single projection
			*b: a single projection
		Output:
			* vectorization of Fundamental Matrix  
		*/
		template <typename VectorType>
		Vector<VectorType> LinVec( const Vector<VectorType>& a, const Vector<VectorType>& b)
		{
			Vector<VectorType> ret(9);
			ret << b(0) * a(0), b(0) * a(1), b(0), b(1) * a(0), b(1) * a(1), b(1), a(0), a(1), 1;
			return ret;
		}

	}

	//A class to deal efficiently with the pyramidal visibility score from "Structure-from-Motion Revisisted", Schonberger & Frahm, CVPR16.
	struct PyramidalVisibilityScore
	{
	public:

		PyramidalVisibilityScore() {};

		//initialize eigen matrices in initializer list
		//initialize a PVS, including width/height/dim_range 
		//projs should be a matrix consisting of 2 rows and an unspecified amount of cols, most likely a submatrix of a larger matrix
		PyramidalVisibilityScore(int width, int height, int level, const MatrixDynamicDense<double>& projs)
			: width_range((int) (pow(2, level) + 1)),	//vector
			height_range((int)(pow(2, level) + 1)),	//vector
			dim_range(level),	//array
			proj_count((int)(pow(2,level)), (int)(pow(2, level))), //2 dimensional SPARSE matrix with this amount of rows/cols
			//probably a lot more than will ever be used
			width(width),height(height), level(level)
		{
			//WIDTH_RANGE
			//filled with range from 0-incerementAmount*valAmount
			width_range = MapRangeToRowVec(width / (pow(2, level)), (int)(pow(2, level) + 1));/*range of values INCLUDING 0*/
	
			//HEIGHT_RANGE
			//filled with range from 0-incerementAmount*valAmount
			height_range = MapRangeToRowVec(height / (pow(2, level)), (int)(pow(2, level) + 1));

			//DIM_RANGE
			//filled with 2^(level-x)
			int current = level;
			std::vector<int> valVec(level);

			std::generate(valVec.begin(), valVec.end(),
				[&current]()->int {current--; return (int)pow(2, current); }
			);
			
			dim_range = EigenHelpers::StdVecToEigenArray(valVec);

			//PROJECTIONS
			if (!AddProjections(projs))
			{
				std::cout << "PVS: Projs Dynamix matrix is empty" << std::endl;
			}
		};

		int GetLevel() { return level; };
		int GetWidth() { return width; };
		int GetHeight() { return height; };

		//Compute the pyramidal visibility score of the current projections, normalized by the maximum score if required
		//returns a double in case the score is normalized
		double ComputeScore(bool normalizeScore = false) 
		{
			double score = 0.0;

			//level is at least one and at least 1 nnz in proj_count
			if (level > 0 && proj_count.nonZeros()!=0)
			{
				//get logical matrix saying whether or not 
				MatrixColSparse<int32_t,int> visible(proj_count);

				//lambda because default prune looks at absolute values of val
				//essentially create a 'logical' matrix, with all existing values being 1
				//visible.prune([](int64_t, int64_t, const int32_t& val) {  
				//	if (val >= 0)
				//	{
				//		return 1;
				//	}
				//	else
				//	{
				//		return 0;
				//	}
				//	});

				//is there a valid reason to keep proj_count around afterwards??
				//maybe perform prune() on proj_count directly instead of a copy?

				std::vector<Eigen::Vector2i> indices;

				//get all indices of points in visible
				for (int k = 0; k < proj_count.outerSize(); ++k)
				{
					for (MatrixColSparse<int32_t,int>::InnerIterator it(proj_count, k); it; ++it)
					{
						indices.push_back(Eigen::Vector2i(it.row(), it.col()));
						//std::cout << "row: " << it.row() << " col: " << it.col() << " val: " << it.value() << std::endl;
					}
				}

				for (size_t i = 0; i < dim_range.size(); i++)
				{
					score += indices.size() * dim_range(i);
					//original matlab code re-created a new matrix of half the size at every iteration
					//instead simply work with the list of indices

					for (size_t j = 0; j < indices.size(); j++)
					{
						indices[j] = Eigen::Vector2i(
							((indices[j][0]) / 2 ),
							((indices[j][1]) / 2 )
							);
					}

					//sort and erase duplicates
					std::sort(indices.begin(), indices.end(), EigenHelpers::PointComparison);
					indices.erase(std::unique(indices.begin(), indices.end(), EigenHelpers::PointEquality), indices.end());

				}

				if (normalizeScore)
				{
					score /= MaxScore();
				}
			}
			
			return score;
		};

		//get the maximum score possible
		int MaxScore()
		{
			if (level > 0)
			{
				//score = sum(this.dim_range .* (this.dim_range * 2) . ^ 2);
				//.* element-wise multiplication
				//.^element wise power
				return (dim_range * (dim_range * 2).pow(2)).sum();
			}
			return 0;
		};

		//add projection to the pyramid
		bool AddProjections(const MatrixDynamicDense<double>& projs)
		{
			if (projs.cols() > 0)
			{

				Eigen::ArrayXi idx_width, idx_height;
				std::tie(idx_width, idx_height)=CellVisible(projs);
		

				//idx_width/idx_height have exact same dimensions
				for (size_t i = 0; i < idx_width.size(); i++)
				{
					//proj_count(idx_height(i), idx_width(i)) = proj_count(idx_height(i), idx_width(i))+1 ; //legacy for dense matrix

					proj_count.coeffRef(idx_height[i]-1, idx_width[i]-1) ++;
				}

				return true;
			}
			
			return false;
		};

		//remove projections from the pyramid
		//functionally almost identical to AddProjections, removing 1 instead of adding it 
		bool RemoveProjections(const MatrixDynamicDense<double>& projs)
		{
			if (projs.cols() > 0)
			{
				Eigen::ArrayXi idx_width, idx_height;

				std::tie(idx_width, idx_height) = CellVisible(projs);

				for (size_t i = 0; i < idx_width.size(); i++)
				{
					//proj_count(idx_height(i), idx_width(i)) = proj_count(idx_height(i), idx_width(i)) - 1; //legacy for dense matrix

					//cannot use tripletlist as using setFromTriplets would override all current values in proj_count
					proj_count.coeffRef(idx_height(i), idx_width(i)) = proj_count.coeff(idx_height(i), idx_width(i)) - 1;
				}


				return true;
			}

			return false;
		}

	private:
		//Compute the indexes of the cells where the projections are visibles
		std::tuple<Eigen::ArrayXi, Eigen::ArrayXi> CellVisible(const MatrixDynamicDense<double> &projs)
		{
			//Arrays filled with ones
			//arrays as opposed to vectors as arrays are more general purpose and allow for more general arithmetic

			//initilaze with 0's instead of 1's as we're working with indices
			//indexing start in 0 with eigen but 1 with matlab

			Eigen::ArrayXi idx_width = Eigen::ArrayXi::Ones(projs.cols());
			Eigen::ArrayXi idx_height = Eigen::ArrayXi::Ones(projs.cols());

			for (size_t i = 0; i < dim_range.size(); i++)
			{
				Eigen::ArrayXi idx_middle = idx_width + dim_range[i]; //middle of the cell

				//possible replacement for current loop??
				//idx_width(j) = ( projs(0, j) > width_range(idx_middle(j)) ).select( idx_middle(j), idx_width(j) );

				//width
				//loop through the arrayXi

				for (size_t j = 0; j < projs.cols(); j++)
				{
					//idx_width(j) = ( projs(0, j) > width_range(idx_middle(j)) ).select( idx_middle(j), idx_width(j) );

					//determine whether points are located to the right or left of idx_middle
					if (projs(0, j) > width_range(idx_middle(j)-1))
					{
						//'move' to the next cell for next lvl/iteration
						idx_width(j) = idx_middle(j);
					}
				}

				//height
				//reuse idx_middle
				idx_middle = idx_height + dim_range[i]; //middle of the cell

				for (size_t j = 0; j < projs.cols(); j++)
				{
				
					//first row instead of second 0 -> Y values
					if (projs(1, j) > height_range(idx_middle(j)-1))
					{
						idx_height(j) = idx_middle(j);
					}
				}
			}

			//these values are not/should not altered afterwards so std::make_tuple() instead of std::tie()
			return std::make_tuple(idx_width, idx_height);
		}

		template <typename T>
		//map a range from [0 - incrementAmount* valAmount] to an Vector<T> by first creating a std::vector<T> 
		//and generating the values in there
		Vector<T> MapRangeToRowVec(const T incrementAmount, const int valAmount)
		{
			T current=(T)0;
			std::vector<T> valVec(valAmount);
		
			std::generate(valVec.begin(), valVec.end(),
				[&current, &incrementAmount]() { T val = current; current += incrementAmount; return val; });
		
			return EigenHelpers::StdVecToEigenVec(valVec);
		}

		int level = 0, width = 0, height = 0;

		//'Vector' is a single row matrices typedef
		Vector<double> width_range, height_range;
		//'Array' typedef as opposed to matrix for better arithmetic options
		Array<int> dim_range;
		
		MatrixColSparse<int32_t,int> proj_count;
	};

	struct Options {
		//default contructor for default param 
		Options() {};

		enum Elimination_Method
		{
			DLT,
			PNV
		};

		enum InformDebug
		{
			NONE,
			REGULAR,
			VERBOSE
		};

		//General options
		Elimination_Method elimination_method = Elimination_Method::DLT; // Method to use for eliminating the projective depths(DLT or PINV)
		int score_level = 6; // Number of levels used in the pyramidal vsibility score
		//eligibility thresholds for views at each level(min visible points),take only this maximum number of views, ordered by the PVS score)
		Eigen::Matrix<int, 2, 9> eligibility_view = (Eigen::Matrix<int,2,9>() << 84, 60, 48, 36, 24, 18, 12, 10, 8, 8, 4, 4, 2, 2, 2, 1, 1, 1).finished();
		int init_level_views = 0; // the initial level of eligibility thresholds for views
		int max_level_views = 4; // the maximum level of eligibility thresholds for views
		int eligibility_point[9] = { 10,9, 8,7, 6, 5, 4, 3, 2 }; // eligibility thresholds for points at each level
		int init_level_points = 6; // the initial level of eligibility thresholds for points
		int max_level_points = 7; // the maximum level of eligibility thresholds for points
		bool differ_last_level = false; // differ the last level of eligibility thresholds for points until last resort situation
		int min_common_init = 200; // Number of minimum common projections for the initial pair
		int max_models = 5; // Number of maximum models, reconstruction is restarted from a completely new pair of views each time

		//Robust estimations
		bool robust_estimation = true;
		int minimal_view[9] = { 10, 10, 9, 9, 9, 8, 8, 8, 7 }; // size of minimal problems for a view at each level
		int minimal_point[9] = { 4,4,4,3,3,3,3,3,2 }; // size of minimal problems for a point at each level
		double outlier_threshold = 2 * 1.96; // reprojection error threshold to consider a projection as an outlier(in pixel)
		double system_threshold = 1e-1; // threshold of the normalized norm of A*x to consider the estimation a failure
		double rank_tolerance = 1e-5; // tolerance on diagonals elements in qr decomposition to estimate rank deficiency
		int max_iter_robust[2] = { 500, 1000 }; // max iterations done in robust estimation
		double confidence = 99.99; // desired confidence in percentage of finding the optimal inlier set to stop early

		//Refinement options
		int max_iter_refine = 50; // max iterations done in refinement
		double min_change_local_refine = 5e-4; // min change to stop local refinement early
		double min_change_global_refine = 3e-4; // min change to stop global refinement early
		int min_iter_refine = 2; // min iterations before stopping early because of min change
		int global_refine = 5; // how much direction change between each global refinement during completion
		bool final_refinement = true; // enable the final refinement
		int max_iter_final_refine = 100; // max iterations done in final refinement
		double min_change_final_refine = 1e-4; // min change to stop final refinement early

		// Merging options
		int merge_min_points = 4 * 10; // Minimum number of common points to attempt a merging
		int merge_min_views = 2 * 10; // Minimum number of common views to attempt a merging

		//metric upgrade options
		int upgrade_threshold = 15; // Threshold used when increasing the test tolerance(20 seems reasonable, 100 maximum)

		// Diagnosis options
		InformDebug debuginform = InformDebug::REGULAR; //NONE,REGULAR,VERBOSE
		bool debug = false; // display everything possible, that's a lot of things
		bool diagnosis = false; // display the graphical diagnosis
		bool diagnosis_upgrade = false; // include the metric upgrade in graphical diagnosis(takes a lot of time !)
		bool diagnosis_completed = false; // include the completed measurements in graphical diagnosis(usually slow)
		bool diagnosis_pause = false; // true = pause indefinetly, false = no pause at all, numerical value = time to pause
		bool diagnosis_save = false; // false = no save, a text string = template to save to(must have a %d field)
		bool diagnosis_cameras = false; // display the cameras in the 3D model
	};

	/*
	Input:
		* measEntry : single homogeneous coordinate
	Output:
		*
		*
	*/
	template <typename MatrixType>
	std::tuple<Eigen::Matrix<MatrixType, 1, 3>, Eigen::Matrix<MatrixType, 2, 3>>
		EliminateDLT(const Eigen::Matrix<MatrixType, 3, 1>& measEntry)
	{
		Eigen::Matrix<MatrixType, 1, 3> cpm_meas = measEntry.transpose() / measEntry.array().pow(2).sum();

		Eigen::Matrix<MatrixType, 2, 3>data;

		data << 0, -measEntry.coeff(2, 0), measEntry.coeff(1, 0),
			measEntry.coeff(2, 0), 0, -measEntry.coeff(0, 0);

		return { cpm_meas, data };
	}

	/*
	Input:
		* measEntry : single homogeneous coordinate
	Output:
		*
		*
	*/
	template <typename MatrixType>
	std::tuple<Eigen::Matrix<MatrixType, 1, 3>, Eigen::Matrix<MatrixType, 2, 3>>
		EliminatePINV(const Eigen::Matrix<MatrixType, 3, 1>& measEntry)
	{
		Eigen::Matrix<MatrixType, 1, 3> pinv_meas = measEntry.transpose() / measEntry.array().pow(2).sum();

		Eigen::Matrix<MatrixType, 3, 3> data = measEntry * pinv_meas;
		data.coeffRef(0, 0) -= (MatrixType)1;
		data.coeffRef(1, 1) -= (MatrixType)1;

		return { pinv_meas, data.block<2,3>(0,0) };
	}


	/*
	Transform the original image coordinates into normalised homogeneous coordinates and various sparse matrices needed.
	 Input:
      * measurements: Original image coordinates (2FxN sparse matrix where missing data are [0;0])
      * options:  Structure containing options, can be left blank for default values
     Output:
	  * measurements: Unnormaliazed homogeneous measurements of the projections, used for computing errors and scores (3FxN
      * data: Matrix containing the data to compute the cost function (2Fx3N)
      * pinv_meas: Matrix containing the data for elimination of projective depths,
	    cross-product matrix or pseudo-inverse of the of the normalized homogeneous coordinates (Fx3N)
      * norm_meas: Normalized homogeneous projections coordinates (2FxN sparse matrix where missing data are [0;0])
      * visible: Visibility DENSE matrix binary mask (FxN)
      * normalisations: Normalisation transformation for each camera stacked vertically (3Fx3)
      */
	template <typename MatrixType, typename IndexType>
	void PrepareData(MatrixColSparse<MatrixType, IndexType>& measurements,
		MatrixColSparse<MatrixType,  IndexType>& data,
		MatrixColSparse<MatrixType,  IndexType>& pinv_meas,
		MatrixColSparse<MatrixType,  IndexType>& norm_meas,
		MatrixColSparse<MatrixType,  IndexType>& normalisations,
		MatrixDynamicDense<bool>& visibility,
		const Options& options = Options())
	{
		//Points not visible enough will never be considered, removing them make data matrices smaller and computations more efficient
		const int optionsVal = options.eligibility_point[options.max_level_points];

		//setup visibility matrix
		visibility.resize(measurements.rows() / 2, measurements.cols());
		visibility.fill(false);

		//[x,y] -> [x,y,1]
		std::vector<Eigen::Triplet<MatrixType, IndexType>> tripletList;
		tripletList.reserve(measurements.nonZeros() / 2);

		IndexType prevCol = 0, prevRow = 0;
		MatrixType prevVal = (MatrixType)0;

		for (int k = 0; k < measurements.outerSize(); ++k)
		{
			for (typename MatrixColSparse<MatrixType, IndexType>::InnerIterator it(measurements, k); it; ++it)
			{
				//chance of data having an x or y equal to 0 is very slim but not impossible
				if (prevRow + 1 == it.row() && prevCol == it.col() )
				{
					if ((it.value() + prevVal) > optionsVal)
					{			
						visibility.coeffRef(it.row() / 2, it.col()) = true;	//replace with triplet
						//[x,y] -> [x,y,1] affine to homogenous
						tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(prevRow + (prevRow / 2), prevCol, prevVal)); //prev it value
						tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(it.row() + (it.row() / 2), it.col(), it.value())); //current it
						tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(it.row() + (it.row() / 2) + 1, it.col(), 1)); //homogenous value (=1)
						++it;
					}
				}
				else 	//in case a a data entry only has an x or y value, not both
				{	
			
				}

				prevCol = it.col();
				prevRow = it.row();
				prevVal = it.value();
			}
		}

		measurements.reserve(tripletList.size());
		measurements.resize(measurements.rows() + (measurements.rows() / 2), measurements.cols());
		measurements.setFromTriplets(tripletList.begin(), tripletList.end());

		const int numVisible = measurements.nonZeros();
		const int numViews = measurements.rows();
		const int numPoints = measurements.cols();


		AlgebraHelpers::NormTrans(measurements, norm_meas, normalisations);


		std::vector<Eigen::Triplet<MatrixType,IndexType>> tripletDLTPINV;	
		std::vector<Eigen::Triplet<MatrixType,IndexType>> tripletData;	
		MatrixType firstVal = (MatrixType)0, secondVal = (MatrixType)0;

		Eigen::Matrix<MatrixType, 1, 3> measRet;
		Eigen::Matrix<MatrixType, 2, 3> dataRet;
		for (int k = 0; k < norm_meas.outerSize(); ++k)
		{
			for (typename MatrixColSparse<MatrixType, IndexType>::InnerIterator it(norm_meas, k); it; ++it)
			{
				const IndexType rowVal = it.row();
				if ((rowVal + 1) % 3 == 0)
				{

					Eigen::Matrix<MatrixType, 3, 1> entry(firstVal, secondVal, it.value());
					if (options.elimination_method == Options::Elimination_Method::DLT)
					{
						std::tie(measRet,dataRet) = EliminateDLT(entry);
					}
					else if (options.elimination_method == Options::Elimination_Method::PNV)
					{
						std::tie(measRet, dataRet) = EliminatePINV(entry);
					}

					//PINV/DLT data
					//add 3 data points to the matrix in following form with 
					//col incrementing based on counter
					//row is it.row/3
					//[cpm/dlt.x cpm/dlt.y cpm/dlt.z]
					tripletDLTPINV.push_back(Eigen::Triplet<MatrixType, IndexType>(rowVal/ 3, it.col() * 3, measRet[0]));
					tripletDLTPINV.push_back(Eigen::Triplet<MatrixType, IndexType>(rowVal/ 3, it.col() * 3 + 1, measRet[1]));
					tripletDLTPINV.push_back(Eigen::Triplet<MatrixType, IndexType>(rowVal/ 3, it.col() * 3 + 2, measRet[2]));

					//insert 'dataRet' in it's current form into a larger sparse matrix
					for (size_t i = 0; i < 6; i++)
					{
						tripletData.push_back(Eigen::Triplet<MatrixType, IndexType>(rowVal/3*2 + (i%2),it.col() * 3 + i/2,dataRet(i)));
					}
				}
				else
				{
					firstVal = secondVal;
					secondVal = it.value();
				}
			}
		}

		pinv_meas.resize(norm_meas.rows()/3, norm_meas.cols()*3);
		pinv_meas.reserve(tripletDLTPINV.size());
		pinv_meas.setFromTriplets(tripletDLTPINV.begin(), tripletDLTPINV.end());

		data.resize(measurements.rows()/3*2 , norm_meas.cols() * 3);
		data.reserve(tripletData.size());
		data.setFromTriplets(tripletData.begin(), tripletData.end());

	}

	/*%
	Compute the pairwise view affinity score.
	 Input:
		* measurement : Unnormaliazed measurements of the projections(image space), used for computing errors and scores(3FxN)
		* visible : Visibility matrix binary mask(FxN)
		* image_sizes Size of the images, used to compute the pyramidal visibility scores(2xF)
		* options : Structure containing options(must be initialized by ppsfm_options to contains all necessary fields)
	 Output :
	    * view_pairs : Pairs of views for which a score has been computed(Kx2)
		* affinity : Affinity score for each pair of views(Kx1)
	*/
	template <typename MatrixType, typename IndexType>
	std::tuple<std::vector<int>, MatrixDynamicDense<int> > 
		PairsAffinity(const MatrixColSparse<MatrixType, IndexType>& measurements,
		const MatrixDynamicDense<bool>& visibility, 
		const MatrixDynamicDense<int>& img_size,
		const Options& options = Options())
	{
		//iterate over the transposed matrix so we go over the matrix row by row
		const MatrixColSparse<MatrixType, IndexType> transposedMat(measurements.transpose());
		
		//const int numViews = visibility.rows();
		const int num_pairs = (visibility.rows()*(visibility.rows() + 1) ) / 2;

		//REPLACE WITH VECTOR<int>??
		//Vector<int> affinity(num_pairs);
		std::vector<int> affinity;
		
		affinity.resize(num_pairs, 2);
		
		MatrixDynamicDense<int> view_pairs(num_pairs, 2);	//keep track of what firstView is being compared with what secondView
		view_pairs.setZero();
		
		int offset = 0;	//a count of how many views have been compared 
		std::vector<Eigen::Triplet<MatrixType, IndexType>> tripletView;
		
		//iterate over all the views (3 rows containing homogeneous coords)
		for (int firstLoop = 0; firstLoop < transposedMat.outerSize()-3; firstLoop+=3)
		{
			//create a second double for-loop to acquire the second view and then compare the first and second view
			for (int secondLoop = firstLoop + 3; secondLoop < transposedMat.outerSize(); secondLoop += 3)
			{
				//get the common points
				auto commonPoints = visibility.row(firstLoop / 3) && visibility.row(secondLoop / 3);
				//how many commonpoints are there?
				if (commonPoints.count() > options.min_common_init)
				{

					//MatrixDynamicDense<MatrixType> firstViewDense(2, visibility.row(firstLoop / 3).array().count());
					//MatrixDynamicDense<MatrixType> secondViewDense(2, visibility.row(secondLoop / 3).array().count());
					MatrixDynamicDense<MatrixType> firstViewDense, secondViewDense;

					std::tie(firstViewDense,secondViewDense) = EigenHelpers::GetViewsCommonPoints(transposedMat, visibility, firstLoop, secondLoop);
			
					//set combo current first/second view in view-pairs
					view_pairs(offset, 0) = firstLoop/3;
					view_pairs(offset, 1) = secondLoop/3;
		
					PyramidalVisibilityScore pvs_first, pvs_second;
		
					if (img_size.cols() == 1)	//single entry in img_size
					{
						pvs_first = PyramidalVisibilityScore(img_size(0), img_size(1), options.score_level, firstViewDense);
						pvs_second = PyramidalVisibilityScore(img_size(0), img_size(1), options.score_level,secondViewDense);
					}
					else
					{
						pvs_first = PyramidalVisibilityScore(img_size.coeff(0, firstLoop), img_size.coeff(1, firstLoop), options.score_level, firstViewDense);
						pvs_second = PyramidalVisibilityScore(img_size(0), img_size(1), options.score_level, secondViewDense);
					
					}
		
					affinity[offset] = pvs_first.ComputeScore(false) + pvs_second.ComputeScore(false);
					offset++;	//increment offset
				}
			}
		
		}


		//reserved more space/initialized more than was needed -> resize
		MatrixDynamicDense<int> pairsResized = view_pairs.block(0, 0, offset,2);
		affinity.resize(offset);

		//sort affinity in descending order -> then reorder view pairs in the same manner
		EigenHelpers::sortVectorAndMatrix(affinity, pairsResized);

		//return affinity and view_pairs
		return {affinity,pairsResized };
	}

	/*
	Estimate, possibly robustly, the fundamental matrix between two images containing matched projections.
	No normalisation is done, the projections matrices should be provided normalized.
	This implementation uses the 8 points algorithm, so at least 8 projections are required.
	
	Input:
		* projs1, the projections in the first image(2xN, N > 8)
		* projs2, the corresponding projections in the second image (2xN)
		* method, method for robust estimation if wanted(currently only support 'ransac', an optimized implementation of RANSAC with MSAC score)
		* confidence, the confidence to stop robust estimation early in RANSAC
	    * max_iter, the maximum number of iterations in RANSAC
	    * dist_thresh, the distance threshold for computing inliers in RANSAC
	Output:
		* fund_mat, the estimated fundamental matrix(3x3)
		* inliers, a binary or index vector indicating the projections used in the estimation
	*/
	template <typename MatrixType>
	std::tuple< MatrixDynamicDense<MatrixType>, VectorVertical<MatrixType>, bool>
		EstimateFundamentalMat(const MatrixDynamicDense<MatrixType>& projs1,
		const MatrixDynamicDense<MatrixType> projs2,
		const EigenHelpers::ESTIMATION_METHOD method = EigenHelpers::ESTIMATION_METHOD::DEFAULT,
		const double confidence = 99.0,
		const int max_iterations = 1000,
		const double distance_threshold = 1e-5,
		const Options& options = Options())
	{
		//make sure projs are of the correct, corresponding formats
		//break here somehow
		MatrixDynamicDense<MatrixType> fundMat(3, 3);
		fundMat.setZero();
		VectorVertical<MatrixType> inliers(projs1.cols());
		inliers.setZero();

		//should not reach here-> do outside of the function
		if (projs1.rows() != 2)
		{
			std::cout << std::endl << "EstimateFundamentalMat: incorrect size for PROJS1 ROWS" << std::endl;
			return { fundMat,inliers,false };

		}
		else if(projs1.cols() < 8)
		{
			std::cout << std::endl << "EstimateFundamentalMat: PROJECTIONS NEED AT LEAST 8 COLUMNS" << std::endl;
			return { fundMat, inliers,false };

		}
		else if (projs1.rows() != projs2.rows() || projs1.cols() != projs2.cols())
		{
			std::cout << std::endl << "EstimateFundamentalMat: The projections matrices must have the same size (2xN)" << std::endl;
			return { fundMat,inliers,false };

		}


		MatrixDynamicDense<MatrixType> coeffs(projs1.cols(),9);
		for (size_t i = 0; i < projs1.cols(); i++)
		{
			coeffs.row(i) = AlgebraHelpers::LinVec((Vector<MatrixType>)projs2.col(i), (Vector<MatrixType>)projs1.col(i));
		}

		switch (method)
		{
		case EigenHelpers::ESTIMATION_METHOD::RANSAC:
		{
			MatrixDynamicDense<MatrixType> fundamentalMat(3, 3);
			std::tie(inliers, fundamentalMat) = AlgebraHelpers::RansacEstimate(coeffs, confidence, max_iterations, distance_threshold);


			//do we have enough inliers (options.min_common_init)
			if (inliers.rows() < options.min_common_init)
			{
				std::cout << std::endl << "EstimateFundamentalMat: not enough inliers found" << std::endl;
				return { fundMat,inliers,false };
			}

			Eigen::JacobiSVD<MatrixDynamicDense<MatrixType>> svd(fundamentalMat, Eigen::ComputeFullU | Eigen::ComputeFullV);

			MatrixDynamicDense<MatrixType> singularMat(2, 2);	//plug singular values in matrix for use in calculations
			singularMat << svd.singularValues()[0], 0,
				0, svd.singularValues()[1];

			MatrixDynamicDense<MatrixType> matV = svd.matrixV().block(0,0,3,2);
			matV.transposeInPlace();
			
			fundMat = svd.matrixU().block(0, 0, 3, 2) * singularMat * 	matV; //should be rank 2
			//2 first values singular values should be a lot higher than 3rd value -> if not return false
			//TODO: IMPLEMENT RANK CHECK IF SINGULAR 

			break; 
		}
		case EigenHelpers::ESTIMATION_METHOD::DEFAULT:
		{
			fundMat = AlgebraHelpers::Estimate(coeffs, true);
			inliers.setOnes();
		}
			break;
		default:
			break;
		}

		// We estimated F_21 instead of F_12 -> transpose
		fundMat.transposeInPlace();
		return {  fundMat,inliers,true };
	}




	template <typename MatrixType>
	struct InitialSolveOutput
	{
		MatrixDynamicDense<MatrixType> cameras;
		MatrixDynamicDense<MatrixType> points;
		MatrixDynamicDense<int> fixed;
		Vector<int> pathway;
	};


	//TODO: ADD A MORE DETAILED DESCRIPTION
	/*
	Retrieve projective Camera position (one of two, with the other being considered an identity) and projective Points positions from fundamental matrix
	Input:
		*
		*
	Output:
		* cameras : Projective estimation of the initial cameras (6x4)
		* points : Projective estimation of the initial points (4xK)
		* pathway : Array containing the order in which views(negative) and points(positive) has been added (1 x 2 + K)
		* fixed : Cell containing arrays of the points or views used in the constraints to add initial views and points (1 x 2 + K)
	*/
	template <typename MatrixType, typename IndexType>
	InitialSolveOutput<MatrixType> ComputeCamPts(
		const MatrixColSparse<MatrixType, IndexType>& norm_meas,
		const Vector<int>& initial_points,
		const MatrixDynamicDense<MatrixType>& fundamental_mat,
		const Eigen::Vector2i initial_views)
	{
		//singular value decomposition
		Eigen::JacobiSVD<MatrixDynamicDense<MatrixType>> svd(fundamental_mat, Eigen::ComputeFullV);
		Eigen::Matrix<MatrixType, 3, 1> epipole = svd.matrixV().block(0, 2, 3, 1) * -1;

		//create 'empty' matrix to store projective depths into
		MatrixDynamicDense<MatrixType> proj_depths(2, initial_points.size());
		proj_depths.setOnes();

		for (size_t i = 0; i < initial_points.size(); i++)
		{
			//get a certain column from second view in sparse norm_meas
			Eigen::Matrix<MatrixType, 3, 1> firstViewCoord = EigenHelpers::GetMeasurementFromView(norm_meas, initial_views(0), initial_points[i]);
			Eigen::Matrix<MatrixType, 3, 1> secondViewCoord=EigenHelpers::GetMeasurementFromView(norm_meas,initial_views(1),initial_points[i]);
			
			Eigen::Matrix<MatrixType, 3, 1> point_epipole = secondViewCoord.cross(epipole);	//cross product value

			//set value in proj_depths
			//0-th index as Eigen cannot verify if the result will be a single scalar at compile time
			proj_depths.coeffRef(1, i) = abs(((firstViewCoord.transpose() * fundamental_mat * point_epipole) / (point_epipole.transpose() * point_epipole))(0));
		}

		//second row of proj_depth/ (sum first 2 elements second row) *2
		proj_depths.row(1) = proj_depths.row(1) / (proj_depths.coeff(1, 0) + proj_depths.coeff(1, 1)) * 2;

		//starting from third measurement (divide cols by their own average)*2
		for (size_t i = 2; i < initial_points.size(); i++)
		{
			proj_depths.col(i) = proj_depths.col(i) / (proj_depths.coeff(0, i) + proj_depths.coeff(1, i)) * 2;
		}

		Vector<int> pathway(initial_points.size() + 2);
		// intial views are added to initial points between element 0-1 and 1-2
		pathway << initial_points(0), -initial_views(0), initial_points(1), -initial_views(1), initial_points.block(0, 2, 1, initial_points.cols() - 2);//rest of initial_points

		MatrixDynamicDense<int> fixed( 2, initial_points.size() + 2);
		fixed.setZero();
		fixed.coeffRef(0, 0) = initial_views(0);
		fixed.coeffRef(0, 1) = initial_points(0);
		fixed.coeffRef(0, 2) = initial_views(0);

		fixed.coeffRef(0, 3) = initial_points(0);
		fixed.coeffRef(1, 3) = initial_points(1);
		for (size_t i = 4; i < fixed.cols(); i++)
		{
			fixed.col(i) = initial_views;
		}

		//get full rows first/second view with all cols in initial points
		MatrixDynamicDense<MatrixType> scaled_meas(6, initial_points.size());;

		for (size_t i = 3*initial_views(0); i < 3 * initial_views(0)+3; i++)
		{
			//only get col id's present in initial points
			//this is possible in the dev branch of Eigen but not in current stable version (3.3)
			//TODO: REPLACE ONCE EIGEN RECEIVES AN UPDATE TO THE STABLE RELEASE
			for (size_t j = 0; j < initial_points.size(); j++)
			{
				//element wise multiplication with the corresponding element in proj_depths
				scaled_meas(i % 3, j) = norm_meas.coeff(i, initial_points[j]) * proj_depths(0,j);
			}
		}

		for (size_t i = 3 * initial_views(1); i < 3 * initial_views(1) + 3; i++)
		{
			for (size_t j = 0; j < initial_points.size(); j++)
			{
				scaled_meas(i % 3+3, j) = norm_meas.coeff(i, initial_points[j]) * proj_depths(1,j);
			}
		}


		Eigen::BDCSVD<MatrixDynamicDense<MatrixType>> svdScaled(scaled_meas, Eigen::ComputeFullU|Eigen::ComputeFullV);

		//create diagonal matrix with first 4 singular values
		Eigen::Matrix<MatrixType, 4, 4> singularDiag = Eigen::Matrix<MatrixType, 4, 1>
		(sqrt(svdScaled.singularValues()(0)), sqrt(svdScaled.singularValues()(1)), sqrt(svdScaled.singularValues()(2)), sqrt(svdScaled.singularValues()(3))).asDiagonal();

		//last two cols flipped signs
		MatrixDynamicDense<MatrixType> cameras = svdScaled.matrixU().block(0, 0, svdScaled.matrixU().rows(), 4) * singularDiag;
		cameras.col(2) *= -1;
		cameras.col(3) *= -1;

		MatrixDynamicDense<MatrixType> points = singularDiag * svdScaled.matrixV().block(0, 0, svdScaled.matrixV().rows(), 4).transpose();

		//points has the last 2 rows flipped due to transposition ( as opposed to cols)
		points.row(2) *= -1;
		points.row(3) *= -1;

		return { cameras,points,fixed,pathway };
	}


	/*
	Find and solve an initial stereo subproblem to start the reconstruction.
		
	Input:
		* norm_meas : Normalized measurements of the projections (3FxN)
		* visible : Visibility matrix binary mask (FxN)
		* view_pairs : Valid pairs of initial images sorted by affinity (Kx2)
		* affinity : Affinity score for the valid pairs of initial images (Kx1)
		* estimated_views : Views that have already been estimated (Kx2)
		* options : Structure containing options, can be left blank to initialise with default values
	Output :
		* tuple containing:
			* cameras : Projective estimation of the initial cameras (6x4)
			* points : Projective estimation of the initial points (4xK)
			* pathway : Array containing the order in which views(negative) and points(positive) has been added (1 x 2 + K)
			* fixed : Cell containing arrays of the points or views used in the constraints to add initial views and points (1 x 2 + K)
		* succes: whether or not valid results could be returned 
	*/
	template <typename MatrixType, typename IndexType>
	std::tuple<InitialSolveOutput<MatrixType>, bool>
		Initialisation(
		const MatrixColSparse<MatrixType, IndexType>& norm_meas,
		const MatrixDynamicDense<bool>& visibility, 
		const std::vector<int>& affinity, 
		const MatrixDynamicDense<int>& view_pairs, 
		const MatrixDynamicDense<int>& estimated_views = MatrixDynamicDense<int>(0,0),
		const Options& options = Options())
	{
		MatrixColSparse<MatrixType, IndexType> transposedMat(norm_meas.transpose());

		//variables for retun value compute_cam_pts
		MatrixDynamicDense< MatrixType> cameras, points;
		MatrixDynamicDense< int> fixed;
		Vector<int> pathway;

		//find matches in view_pairs and estimated_views

		//TO-DO: IMPLEMENT FIND MATCHES
		//estimated_pairs = ismember(view_pairs, estimated_views);    %find matching rows
		
		//both_unestimated = all(~estimated_pairs, 2);        %both row values are 0 / 1
		//one_estimated = any(estimated_pairs, 2) & ~all(estimated_pairs, 2);
		//sorted_idx = horzcat(find(both_unestimated)', find(one_estimated)');


		//match row in estimated_views and view_pairs
		//find matches 
		for (size_t i = 0; i < view_pairs.rows(); i++)
		{
			int firstViewRow = view_pairs(i, 0);
			int secondViewRow = view_pairs(i, 1);

			//get visible points norm meas for both first and second views and estimate fundamental mat from that
			MatrixDynamicDense<MatrixType> firstViewDense, secondViewDense;
			std::tie(firstViewDense,secondViewDense) = EigenHelpers::GetViewsCommonPoints(transposedMat, visibility, 3 * firstViewRow, 3 * secondViewRow);

			VectorVertical<MatrixType> inliers;
			MatrixDynamicDense<MatrixType> fundMat;

			//no easy way to quit out of EstimFundMat when already in -> condition here??
			if (firstViewDense.rows() == 2
				&& firstViewDense.cols() >= 8
				&& (firstViewDense.rows() == secondViewDense.rows() || firstViewDense.cols() == secondViewDense.cols()))
			{
				bool succesfulEstimation;
				std::tie(fundMat,inliers, succesfulEstimation) = 
					EstimateFundamentalMat(firstViewDense, secondViewDense, EigenHelpers::ESTIMATION_METHOD::RANSAC, 99.99, 1000, 1e-3,options);

				//estimate fund mat can fail, if so continue to next iteration loop
				//only need to be succesfull once?
				if (succesfulEstimation)
				{
					//get inliers of visible points
					auto visiblePoints = visibility.row(firstViewRow) && visibility.row(secondViewRow);

					//get the id of all rows that are both in visible points that are also present in inliers inliers
					Vector<int> init_points((inliers.array() != 0).count());

					int count = 0, newId = 0;
					for (size_t j = 0; j < visiblePoints.cols(); j++)
					{
						if (visiblePoints(j) == 1)
						{
							if (inliers(count) != (MatrixType)0)
							{
								init_points.coeffRef(newId) = j;
								newId++;
							}
							count++;
						}
					}

					//std::tie(cameras, points, fixed, pathway) = ComputeCamPts(norm_meas, init_points, fundMat, Eigen::Vector2i(firstViewRow, secondViewRow));
					//return {  cameras,points,fixed, pathway , true };
					return { ComputeCamPts(norm_meas, init_points, fundMat, Eigen::Vector2i(firstViewRow, secondViewRow)) , true };
				}
			}
		}
		
		std::cout << "views did not meet minimum requirements, no good computeCamPts found!, returning empty variables" << std::endl;

		//if reached here, should be empty all outputs are empty -> return false
		return { {cameras,points,fixed,pathway},false };

	}

	template <typename MatrixType, typename IndexType>
	bool CheckExpandInit(
		InitialSolveOutput<MatrixType>& cam_point_locations,
		MatrixColSparse<MatrixType, IndexType>& inliers,
		const int num_views,
		const int num_points)
	{
		//assert(isrow(pathway), 'Initial pathway should be a row');	//inherant to vector<int> format
		//assert(iscell(fixed) && isrow(fixed), 'Initial fixed constraints should be a row of cells');
		//assert(length(pathway) == length(fixed), 'Initial pathway and fixed constraints should have the same lengths');

		if (cam_point_locations.pathway.size() != cam_point_locations.fixed.cols())
		{
			std::cout << "Initial pathway and fixed constraints should have the same lengths" << std::endl;
			return false;
		}

		if (cam_point_locations.cameras.rows() != 3 * num_views)
		{
			MatrixDynamicDense<MatrixType> newCameras(3*num_views,4);
			newCameras.setZero();

			//pathway(1) is element -initial_views(0) back in EstimateFunMat()
			for (size_t i = 0 ; i < 3; i++)
			{
				newCameras.row((-3 * cam_point_locations.pathway(1)) +i + 1) = cam_point_locations.cameras.row(i);
			}

			//pathway(3) is element -initial_views(0) back in EstimateFunMat()
			for (size_t i = 0; i < 3; i++)
			{
				newCameras.row((-3 * cam_point_locations.pathway(3)) + i + 1) = cam_point_locations.cameras.row(i  +3);
			}

			cam_point_locations.cameras.resize(3 * num_views, 4);	//can only resize afterwards to prevent being destructive
			cam_point_locations.cameras = newCameras;
		}



		if (cam_point_locations.points.cols() != num_points)
		{
			MatrixDynamicDense<MatrixType> tempPoints(4, num_points);	//should this be a sparse matrix?
			tempPoints.setZero();

			int count = 0;
			for (size_t i = 0; i < cam_point_locations.pathway.size(); i++)
			{
				//do not consider the 2 initial_views values
				if (cam_point_locations.pathway(i) > 0)
				{
					tempPoints.col(cam_point_locations.pathway(i)) = cam_point_locations.points.col(count);	//issue here
					count++;
				}
			}

			cam_point_locations.points.resize(4, num_points);
			cam_point_locations.points = tempPoints;
		}


		//2 negative initial_views values + inliers needs to be empty still
		//fill inliers sparse matrix with values
		if (inliers.nonZeros() == 0 && (cam_point_locations.pathway.array() < 0).count() == 2)
		{
			std::vector<Eigen::Triplet<MatrixType, IndexType>> tripletList;
			tripletList.reserve(cam_point_locations.pathway.size() * 2);

			const int firstRow = -cam_point_locations.pathway(1);
			const int secondRow = -cam_point_locations.pathway(3);

			//set the two rows for initial_views values to 1 where the col value 
			for (size_t i = 0; i < cam_point_locations.pathway.size(); i++)
			{
				if (cam_point_locations.pathway(i) > 0)
				{
					tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(firstRow, cam_point_locations.pathway(i), 1));
					tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(secondRow, cam_point_locations.pathway(i), 1));
				}
			}

			inliers.setFromTriplets(tripletList.begin(), tripletList.end());
		}		


		const int last_path = cam_point_locations.pathway.cols();

		//conservativeresize() seems to give better performance here, 
		//seeing as we just append points to the end and do not change locations of existing points
		cam_point_locations.pathway.conservativeResize(num_views + num_points);
		cam_point_locations.pathway.block(0, last_path, 1, num_views + num_points - last_path).setZero();

		//resize fixed in a similar manner
		cam_point_locations.fixed.conservativeResize(2,num_views+num_points);
		cam_point_locations.fixed.block(0, last_path, 2, num_views + num_points - last_path).setZero();

		return true;
	}

	/*
	  Display some information allowing diagnosis of the reconstruction given in cameras and points.
	  Depending on the options, this is mainly some graphical figure, but some console text output can also be included.
	  This diagnosis can also be saved to a file which can be reloaded later by calling this functions with only the filename.

	  Input:
		* inliers : Inliers matrix binary mask(FxN)
		* visible : Visibility matrix binary mask(FxN)
		* InitialSolveOutput init_output:
			* cameras : Projective estimation of the cameras that has been completed(3Fx4)
			* points : Projective estimation of the points that has been completed(4xN)
			* pathway : Array containing the order in which views(negative) and points(positive) has been added(1 x k, k <= F + N)
		* normalisations : Normalisation transformation for each camera stacked vertically(3Fx3)
		* img_measurements : Original image measurements of the projections, used for computing reprojection errors(3FxN)
		* centers : Principal points coordinates for each camera(2xF)
		* iter : Iteration number when called from the completion loop
			* options : Structure containing options(must be initialized by ppsfm_options to contains all necessary fields)
		* fig_base : Numeric handle to use for the first figure, next ones will increments this*/
	template <typename MatrixType, typename IndexType>
	void Diagnosis(
		const MatrixColSparse<MatrixType, IndexType>& inliers,
		const MatrixDynamicDense<bool>& visibility,
		const InitialSolveOutput<MatrixType>& init_output,
		const MatrixColSparse<MatrixType, IndexType>& normalisations,
		const MatrixColSparse<MatrixType, IndexType>& img_meas,
		const MatrixDynamicDense<MatrixType>& centers,
		const Options& options = Options(),
		const int fig_base = 0
	)
	{

	}


		//TODO:
		//BETTER DESCRIPTION
		std::tuple<Vector<double> , Vector<int>>
		SearchEligibleViews(
		const Eigen::Vector2i& thresholds,
		const MatrixDynamicDense<bool>& visibility,
		const Vector<int>& known_points,
		const Vector<int>& known_views,
		Vector<int>& rejected_views,
		std::vector<PyramidalVisibilityScore>& pvs_scores)
	{
		const int num_views = visibility.rows();

		Vector<bool> unknown_views(num_views);
		unknown_views.setOnes();	//which views do we not have info on right now??
		for (size_t i = 0; i < known_views.size(); i++)	//should only be 2 elements but do this to be sure
		{
			unknown_views(known_views(i)) = false;
		}

		//number of visible points per view, considering only 'known' points
		Vector<int> num_visible_points(num_views);
		num_visible_points.setZero();

		for (size_t i = 0; i < num_visible_points.size(); i++)
		{
			if (unknown_views(i) == true)
			{
				int count = 0;
				for (size_t j = 0; j < known_points.size(); j++)
				{
					if (visibility(i, known_points(j)) == true)
					{
						count++;
					}
				}
				num_visible_points(i) = count;
			}
			//TODO: UPDATE EIGEN TO 3.4 ON RELEASE, ALLOWING TO GET LIST OF INDICES FROM MATRIX
		}


		//is there enough data per view to consider further testing??
		for (size_t i = 0; i < unknown_views.size(); i++)
		{
			if (num_visible_points(i) <= rejected_views(i) ||
				num_visible_points(i) < thresholds(0))
			{
				unknown_views(i) = false;
			}
		}


		//compute scores for views that are considered eligible
		std::vector<double> scores(num_views,0);
		scores.reserve(num_views);
		//Vector<double> scores(num_views);
		MatrixDynamicDense<int> eligibles(1, num_views);
		Vector<int> eligibles_compressed(1, num_views);
		int count = 0;
		for (int i = 0; i < pvs_scores.size(); i++)
		{
			if (unknown_views(i) == true)
			{
				scores[i] = pvs_scores[i].ComputeScore(true) * 100.0;
				
				eligibles(0,i) = i;
				eligibles_compressed(count) = i;
				count++;
			}
			else
			{
				eligibles(0,i) = -1; //invalid values
			}
		}


		//all scores are returned so we can easily identify the corresponding view with eligibles
		if ( count > thresholds(1))
		{
			//too much eligibles, get best x number 
			//sort scores and get best id's
			EigenHelpers::sortVectorAndMatrix(scores, eligibles,false);
			auto scoresVec = EigenHelpers::StdVecToEigenVec(scores);
			return { scoresVec,eligibles.block(0,0,1,thresholds(1)) };

		}
		else
		{
			//get id's of eligible
			eligibles_compressed.conservativeResize(count);
			auto scoresVec = EigenHelpers::StdVecToEigenVec(scores);
			return { scoresVec,eligibles.block(0,0,1,thresholds(1)) };
		}
	}

	/*
	   Complete the initial factorization provided in cameras and points under the constraint
	   defined by the pathway and fixed entries.
	   Use only the visible entries for which measurements are given as there pseudo inverse(pinv_meas) and a data matrix.
	   Measurements have also been normalized and original measurements are given(for the robust estimation).
	
	 Input:
		* data : Matrix containing the data to compute the cost function(2Fx3N)
		* pinv_meas : Matrix containing the data for elimination of projective depths,
		  cross - product matrix or pseudo - inverse of the of the normalized homogeneous coordinates(Fx3N)
		* visibility : Visibility matrix binary mask(FxN)
		
		* normalisations : Normalisation transformation for each camera stacked vertically(3Fx3)
		* img_meas : Unnormaliazed measurements of the projections, used for computing errors and scores(3FxN)
		* img_sizes : Size of the images, used to compute the pyramidal visibility scores(2xF)
		* centers : Principal points coordinates for each camera(2xF)
		* cam_point_locations: struct containing the struct return value from P2SFM::Initialisation()
			* cameras : Initial cameras estimation(3Fx4 or 3kx4 if k cameras in the initial sub - problem)
			* points : Initial points estimation(4xN or 4xk if k points in the initial sub - problem)
			* fixed : Cell containing arrays of the points or views used in the constraints to add initial views and points(1 x F + N)
			* pathway : Array containing the order in which views(negative) and points(positive) has been added(1 x k, k <= F + N)
		* options : Structure containing options(must be initialized by ppsfm_options to contains all necessary fields)
	 Output :
		* cameras : Projective estimation of the cameras that has been completed(3Fx4)
		* points : Projective estimation of the points that has been completed(4xN)
		* pathway : Array containing the order in which views(negative) and points(positive) has been added(1 x k, k <= F + N)
		* fixed : Cell containing arrays of the points or views used in the constraints when adding views and points(1 x F + N)
		* inliers : Inliers matrix binary mask(FxN)
	*/
	template <typename MatrixType, typename IndexType>
	void Complete(
		const MatrixColSparse<MatrixType, IndexType>& data,
		const MatrixColSparse<MatrixType, IndexType>& pinv_meas,
		const MatrixDynamicDense<bool>& visibility,
		const MatrixColSparse<MatrixType, IndexType>& normalisations,
		const MatrixColSparse<MatrixType, IndexType>& img_meas,
		const MatrixDynamicDense<MatrixType>& img_sizes,
		const MatrixDynamicDense<MatrixType>& centers,
		InitialSolveOutput<MatrixType>& init_output,
		const Options& options = Options()
		)
	{
		const int num_views = visibility.rows();
		const int num_points = visibility.cols();

		//allocate space for a sparse amt
		MatrixColSparse<MatrixType, IndexType> inliers(num_views, num_points);
		inliers.reserve((visibility.array() >0).count());

		CheckExpandInit(init_output,inliers, num_views,num_points);

		//variables used to limit refinement to latest added views/points
		int init_refine = 0;
		int new_init_refine = 0;
		Eigen::Vector2i prev_last_path = Eigen::Vector2i( 0,0 );
		bool last_dir_change_view = 1; //Last direction change toward : 1 = views, 0 = points


		//Number of projection last time each view/point has been rejected
		Vector<int> rejected_views(num_views);
		rejected_views.setZero();
		Vector<int> rejected_points(num_points);
		rejected_points.setZero();

		//PVS for views
		//old init_pvs function in ppsfm_complete
		std::vector<PyramidalVisibilityScore> pvs_scores(num_views);
		MatrixColSparse<MatrixType, IndexType> copyMat(img_meas.transpose());

		//get all elements in pathway that are >0
		Vector<int> copyPath((init_output.pathway.array() > 0).count());
		copyPath << init_output.pathway(0), init_output.pathway(2), init_output.pathway.block(0, 4, 1, (init_output.pathway.array()>0).count() - 2);

		//create matrix for projs, massively oversized (create afterwards without having to retrieve all elements twice??)
		MatrixDynamicDense<MatrixType> projs(2, copyPath.size()); //max possible size

		for (size_t i = 0; i < num_views; i++)
		{

			int counter=0;
			for (size_t j = 0; j < copyPath.size(); j++)
			{
				if (visibility(i, copyPath(j)) == 1)
				{
					//std::cout << img_meas.coeff(3 * i, copyPath(j)) << std::endl;
					//std::cout << img_meas.coeff(3 * i + 1, copyPath(j)) << std::endl;
					projs(0, counter) = img_meas.coeff(3 * i, copyPath(j));
					projs(1, counter) = img_meas.coeff(3 * i + 1, copyPath(j));
					counter++;
				}
			}

			if (img_sizes.cols()==1)
			{
				pvs_scores[i] = PyramidalVisibilityScore(img_sizes(0, 0), img_sizes(1, 0), options.score_level,projs.block(0, 0, 2, counter));
			}
			else
			{
				pvs_scores[i] = PyramidalVisibilityScore(img_sizes(0, i), img_sizes(1, i), options.score_level, projs.block(0, 0, 2, counter));
			}
		}


		//eligibility threshold levels
		bool level_changed = false;
		int level_views = std::max(options.init_level_views, options.max_level_views);
		int level_points = std::max(options.init_level_points, options.max_level_points);

		//get amount of values in pathway <0
		int num_known_views = (init_output.pathway.array() < 0).count();
		int num_known_points = (init_output.pathway.array() > 0).count();	//>0 to avoid empty values appended at the end
		int num_added_views = num_known_points, num_added_points = num_known_points;

		int num_iter = 0;
		int iter_refine = 0;

		//ppsfm_diagnosis()

		//main loop 
		while ((num_known_points < num_points || num_known_views < num_views) &&
			(level_changed || num_added_views + num_added_points >0) )
		{
			level_changed = false;
			num_added_points = 0;
			num_added_views = 0;

			//process views
			if (num_known_views < num_views)
			{
				//search_eligible_views
				Vector<int> eligibles;
				Vector<double> scores;
				std::tie(scores,eligibles)= SearchEligibleViews(options.eligibility_view.col(level_views), visibility, copyPath, Eigen::Vector2i(-init_output.pathway(1),-init_output.pathway(3)), rejected_views, pvs_scores);
				int x = 10;
			}
		}
	}

}
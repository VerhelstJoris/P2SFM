#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <algorithm>
#include <numeric>

//typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixDynamicDense;
//dynamic sized dense matrix template 
template <typename Type> using MatrixDynamicDense = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

//typedef Eigen::SparseMatrix<double, Eigen::ColMajor, int64_t> MatrixColSparse;
//dynamic sized sparse matrix template
template <typename ScalarType, typename IndexType> using MatrixColSparse = Eigen::SparseMatrix<ScalarType, Eigen::ColMajor, IndexType>;

//vector Template
template <typename Type> using Vector = Eigen::Matrix<Type,1, Eigen::Dynamic>;
//array template
template <typename Type> using Array = Eigen::Array<Type, 1, Eigen::Dynamic>;


namespace P2SFM 
{
	namespace EigenHelpers
	{
		//a vector is sorted in descending order, the matrix cols are then are arranged in the same manner as the vector was sorted
		template <typename VecType, typename MatrixType>
		void sortVectorAndMatrix(std::vector<VecType>&vec, MatrixDynamicDense<MatrixType>& mat) 
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
			for (size_t i = 0; i < mat.cols(); i++)
			{
				mat.col(i).swap(temp.col(i));
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
					tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(i, i + 2, -tempArr[i] * scaleArr[i/3]));
					tripletList.push_back(Eigen::Triplet<MatrixType, IndexType>(i+1, i + 2, -tempArr[i] * scaleArr[i/3]));
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
		int ComputeScore(bool normalizeScore = false) 
		{
			int score = 0;

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
		int init_level_views = 1; // the initial level of eligibility thresholds for views
		int max_level_views = 5; // the maximum level of eligibility thresholds for views
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
		*measEntry : single homogeneous coordinate
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
		*measEntry : single homogeneous coordinate
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
		MatrixColSparse<MatrixType, IndexType> transposedMat(measurements.transpose());
		
		//const int numViews = visibility.rows();
		const int num_pairs = (visibility.rows()*(visibility.rows() + 1) ) / 2;

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
					tripletView.clear();
					tripletView.reserve(visibility.row(firstLoop / 3).array().count());

					//get the FIRST view (views passed to PVS are only the first 2 entries of homogeneous coord)
					//THIS CAN MOST LIKELY BE MOVED OUTSIDE OF THIS LOOP -> MASK THIS MATRIX USING EIGEN??
					int colOffset = 0;
					for (size_t i = firstLoop; i < firstLoop + 2; i++)
					{
						colOffset = 0;

						//k will give data from a single column, construct submatrix from first 3 cols
						for (typename MatrixColSparse<MatrixType, IndexType>::InnerIterator firstIt(transposedMat, i); firstIt; ++firstIt)
						{
							//check if corresponding value in 'commonpoints' is true/false
							if (commonPoints(firstIt.row()) == 1)
							{
								//tripletView.push_back(Eigen::Triplet<MatrixType, IndexType>(firstIt.col() % 3, firstIt.row(), firstIt.value()));
								tripletView.push_back(Eigen::Triplet<MatrixType, IndexType>(firstIt.col() % 3, colOffset, firstIt.value()));
								colOffset++;
							}
						}
					}

					MatrixColSparse<MatrixType, IndexType> firstView(2, colOffset);

					//create firstView
					//firstView.resize(2, colOffset);
					firstView.setFromTriplets(tripletView.begin(), tripletView.end());

					tripletView.clear();
					tripletView.reserve(visibility.row(firstLoop / 3).array().count());

					//get the SECOND view (views passed to PVS are only the first 2 entries of homogeneous coord)
					for (size_t j = secondLoop; j < secondLoop + 2; j++)	//combine this and iterator loop?
					{
						colOffset = 0;

						//k will give data from a single column, construct submatrix from first 3 cols
						for (typename MatrixColSparse<MatrixType, IndexType>::InnerIterator secondIt(transposedMat, j); secondIt; ++secondIt)
						{
							//check if corresponding value in 'commonpoints' is true/false
							if (commonPoints(secondIt.row()) == 1)
							{
								//tripletView.push_back(Eigen::Triplet<MatrixType, IndexType>(secondIt.col() % 3, secondIt.row(), secondIt.value()));
								tripletView.push_back(Eigen::Triplet<MatrixType, IndexType>(secondIt.col() % 3, colOffset, secondIt.value()));
								colOffset++;
							}
						}
				
					}
					MatrixColSparse<MatrixType, IndexType> secondView(2, colOffset);


					//create secondView
					//secondView.resize(2, colOffset);
					secondView.reserve(tripletView.size());
					secondView.setFromTriplets(tripletView.begin(), tripletView.end());


					//set combo current first/second view in view-pairs
					view_pairs(offset, 0) = firstLoop/3;
					view_pairs(offset, 1) = secondLoop/3;
		
					PyramidalVisibilityScore pvs_first, pvs_second;
		
					if (img_size.cols() == 1)	//single entry in img_size
					{
						//std::cout << img_size(0) << " "<<img_size(1) << " "  << options.score_level << std::endl;
						//std::cout << Eigen::MatrixXd(firstView) << std::endl;
						pvs_first = PyramidalVisibilityScore(img_size(0), img_size(1), options.score_level, Eigen::MatrixXd(firstView));
						pvs_second = PyramidalVisibilityScore(img_size(0), img_size(1), options.score_level, Eigen::MatrixXd(secondView));
					}
					else
					{
						pvs_first = PyramidalVisibilityScore(img_size.coeff(0, firstLoop), img_size.coeff(1, firstLoop), options.score_level, Eigen::MatrixXd(firstView));
						pvs_second = PyramidalVisibilityScore(img_size(0), img_size(1), options.score_level, Eigen::MatrixXd(secondView));
					}
		
					affinity[offset] = pvs_first.ComputeScore(false) + pvs_second.ComputeScore(false);
					offset++;	//increment offset
				}
			}
		
		}


		//reserved more space/initialized more than was needed -> resize
		MatrixDynamicDense<int> pairsResized = view_pairs.block(0, 0, offset,2);
		affinity.resize(offset);

		std::cout << "VIEW PAIRS: " << pairsResized << std::endl;

		//std::cout << "AFFINITY: " << std::endl;
		//
		//for (size_t i = 0; i < affinity.size(); i++)
		//{
		//	std::cout << affinity[i] << " ";
		//}

		//sort affinity in descending order -> then reorder view pairs in the same manner
		EigenHelpers::sortVectorAndMatrix(affinity, pairsResized);

		//return affinity and view_pairs
		return {affinity,view_pairs};
	}

	/*
	Find and solve an initial stereo subproblem to start the reconstruction.
		
		   Input:
		 * norm_meas : Normalized measurements of the projections(3FxN)
		 * visible : Visibility matrix binary mask(FxN)
		 * view_pairs : Valid pairs of initial images sorted by affinity(Kx2)
		 * affinity : Affinity score for the valid pairs of initial images(Kx1)
		 * estimated_views : Views that have already been estimated
		     * options : Structure containing options(must be initialized by ppsfm_options to contains all necessary fields)
		 Output :
		     * cameras : Projective estimation of the initial cameras(6x4)
		 * points : Projective estimation of the initial points(4xK)
		 * pathway : Array containing the order in which views(negative) and points(positive) has been added(1 x 2 + K)
		 * fixed : Cell containing arrays of the points or views used in the constraints to add initial views and points(1 x 2 + K)
		 */
	template <typename MatrixType, typename IndexType>
	void Initialisation(
		MatrixColSparse<MatrixType, IndexType> norm_meas,
		const MatrixDynamicDense<bool>& visibility, 
		std::vector<int> affinity, 
		MatrixDynamicDense<int> view_pairs, 
		/*estimated_views,*/
		const Options& options = Options())
	{
	
	}
}
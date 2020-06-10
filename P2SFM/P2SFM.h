#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixDynamicDense;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor, int64_t> MatrixSparse;

//vector Template
template <typename Type> using Vector = Eigen::Matrix<Type, Eigen::Dynamic, 1>;


namespace P2SFM 
{
	namespace EigenHelpers
	{
		template <typename T>
		Vector<T> StdVecToRowVec(std::vector<T> vec)
		{
			T* ptr = &vec[0];
			Eigen::Map<Vector<T>> map(ptr, vec.size());
			return (Vector<T>)map;
		}
	}


	//A class to deal efficiently with the pyramidal visibility score from "Structure-from-Motion Revisisted", Schonberger & Frahm, CVPR16.
	struct PyramidalVisibilityScore
	{
	public:

		//initialize eigen matrices in initializer list
		//initialize a PVS, including width/height/dim_range 
		//projs should be a matrix consisting of 2 rows and an unspecified amount of cols, most likely a submatrix of a larger matrix
		PyramidalVisibilityScore(int width, int height, int level, const MatrixDynamicDense& projs)
			: width_range((int) (pow(2, level) + 1)), height_range((int)(pow(2, level) + 1)), dim_range(level), 
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
			
			dim_range = EigenHelpers::StdVecToRowVec(valVec);

			//Projections
			if (!AddProjections(projs))
			{
				std::cout << "PVS: Projs Dynamix matrix is empty" << std::endl;
			}
		};

		int GetLevel() { return level; };
		int GetWidth() { return width; };
		int GetHeight() { return height; };

		void ComputeScore() { std::cout << dim_range; };

		//get the maximum score possible
		void MaxScore()
		{

		};

		//add projection to the pyramid
		bool AddProjections(const MatrixDynamicDense& projs)
		{
			if (projs.cols() > 0)
			{
				CellVisible(projs);

				return true;
			}
			
			return false;
		};


		void RemoveProjections();

	private:
		//Compute the indexes of the cells where the projections are visibles
		std::tuple<Eigen::ArrayXi, Eigen::ArrayXi> CellVisible(const MatrixDynamicDense &projs)
		{
			//Arrays filled with ones
			//arrays as opposed to vectors as arrays are more general purpose
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
					if (projs(0, j) >= width_range(idx_middle(j)))
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
					//idx_width(j) = ( projs(0, j) > width_range(idx_middle(j)) ).select( idx_middle(j), idx_width(j) );
				
					//first row instead of second 0 -> Y values
					if (projs(1, j) >= width_range(idx_middle(j)))
					{
						idx_height(j) = idx_middle(j);
					}
				}
			}
			return std::make_tuple(idx_width, idx_height);
		}

		template <typename T>
		//map a range from [0 - incrementAmount* valAmount] to an Vector<T> by first creating a std::vector<T> and generating the values in there
		Vector<T> MapRangeToRowVec(const T incrementAmount, const int valAmount)
		{
			T current=(T)0;
			std::vector<T> valVec(valAmount);
		
			std::generate(valVec.begin(), valVec.end(),
				[&current, &incrementAmount]() { T val = current; current += incrementAmount; return val; });
		
			return EigenHelpers::StdVecToRowVec(valVec);
		}

		int level = 0, width = 0, height = 0;

		//'Vector' is a single row matrices typedef
		Vector<double> width_range, height_range;
		Vector<int> dim_range;

		//proj_count = [];
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
		int max_level_points = 8; // the maximum level of eligibility thresholds for points
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
	Transform the original image coordinates into normalised homogeneous coordinates and various sparse matrices needed.
	 Input:
      * orig_meas: Original image coordinates (2FxN sparse matrix where missing data are [0;0])
      * options:  Structure containing options, can be left blank for default values
     Output:
      * data: Matrix containing the data to compute the cost function (2Fx3N)
      * pinv_meas: Matrix containing the data for elimination of projective depths,
	    cross-product matrix or pseudo-inverse of the of the normalized homogeneous coordinates (Fx3N)
      * norm_meas: Normalized homogeneous projections coordinates (2FxN sparse matrix where missing data are [0;0])
      * visible: Visibility matrix binary mask (FxN)
      * normalisations: Normalisation transformation for each camera stacked vertically (3Fx3)
      * img_meas: Unnormaliazed homogeneous measurements of the projections, used for computing errors and scores (3FxN)*/
	void PrepareData(MatrixSparse& measurements, const Options& options = Options())
	{
		//Points not visible enough will never be considered, removing them make data matrices smaller and computations more efficient

	}

	void EliminatePINV()
	{

	}
}
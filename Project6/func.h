#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include "mykmeans.h"
#include "mysvm.h"
#pragma warning(disable:4996);
using namespace cv;
using namespace std;

const int DEFAULT_LABEL_VALUE = -1;
enum { DISTANCE_TYPE_EUCLIDEAN };
enum { CLUSTERING_FLANN, CLUSTERING_KMEANS };
enum { MATCHING_FLANN, MATCHING_EUCLIDEAN };


//特征点类
class IPoint :public Point2f
{
public:
	//float x;
	//float y;
	float dx;
	float dy;
	float scale;
	float orientation;
	float laplacian;
	float descriptor[64];
	float operator-(const IPoint& rhs);
};

//图像类
class IntegralImg
{
public:
	int Width;		//图片的宽
	int Height;		//图片的高
	Mat Original;	//原始图片
	Mat Integral;	//积分图像
	IntegralImg(Mat img);
	float AreaSum(int x, int y, int dx, int dy);
};

//响应层类
class ResponseLayer
{
public:
	//本层图像的宽度
	int Width;
	//本层图像的高度
	int Height;
	//模板作用的步长
	int Step;
	//模板的长度的1/3
	int Lobe;
	//Lobe*2-1
	int Lobe2;
	//模板的长度一半，边框
	int Border;
	//模板长度
	int Size;
	//模板元素个数
	int Count;
	//金字塔级数
	int Octave;
	//金字塔层数
	int Interval;
	//高斯卷积后的图片
	Mat* Data;
	//Laplacian矩阵
	Mat* LapData;

	ResponseLayer(IntegralImg* img, int octave, int interval);
	void BuildLayerData(IntegralImg* img);
	float GetResponse(int x, int y, int step);
	float GetLaplacian(int x, int y, int step);
};

//快速Hessian矩阵类
class FastHessian
{
public:

	IntegralImg Img;
	//图像堆的组数
	int Octaves;
	//为图像堆中每组中的中间层数，该值加2等于每组图像中所包含的层数
	int Intervals;
	//Hessian矩阵行列式响应值的阈值
	float Threshold;

	map<int, ResponseLayer*> Pyramid;
	//特征点矢量数组
	vector<IPoint> IPoints;
	//构造函数
	FastHessian(IntegralImg iImg, int octaves, int intervals, float threshold);
	void GeneratePyramid();
	void GetIPoints();
	void ShowIPoint();
	bool IsExtremum(int r, int c,
		int step, ResponseLayer* t, ResponseLayer* m, ResponseLayer* b);
	void InterpolateExtremum(int r, int c, int step,
		ResponseLayer* t, ResponseLayer* m, ResponseLayer* b);
	void InterpolateStep(int r, int c, int step,
		ResponseLayer* t, ResponseLayer* m, ResponseLayer* b,
		double* xi, double* xr, double* xc);
	Mat Deriv3D(int r, int c, int step,
		ResponseLayer* t, ResponseLayer* m, ResponseLayer* b);
	Mat Hessian3D(int r, int c, int step,
		ResponseLayer* t, ResponseLayer* m, ResponseLayer* b);
};


//surf描述类
class SurfDescriptor
{
public:
	IntegralImg& Img;
	std::vector<IPoint>& IPoints;

	void GetOrientation();
	void GetDescriptor();

	float gaussian(int x, int y, float sig);
	float gaussian(float x, float y, float sig);
	float haarX(int row, int column, int s);
	float haarY(int row, int column, int s);
	float getAngle(float X, float Y);
	float RotateX(float x, float y, float si, float co);
	float RotateY(float x, float y, float si, float co);
	int fRound(float flt);
	void DrawOrientation();

	SurfDescriptor(IntegralImg& img, std::vector<IPoint>& iPoints);
};


//surf使用类
class Surf
{
public:
	Mat inputmat;

	vector<IPoint> GetAllFeatures(Mat img);
};



class FeatureVector
{
public:
	FeatureVector();
	~FeatureVector();
	FeatureVector(int _size)
	{
		resize(_size);
	}
	FeatureVector(const std::vector<float>& _data)
	{
		set(_data);
	}
	FeatureVector(const FeatureVector& cpy);

	void set(const std::vector<float>& _data)
	{
		data = _data;
		size = data.size();
	}
	void get(std::vector<float>& _data)
	{
		_data = data;
	}
	size_t getSize()
	{
		return size;
	}
	bool empty()
	{
		return !size;
	}
	void resize(size_t _size)
	{
		size = _size;
		data.resize(size);
	}
	void clear()
	{
		data.clear();
	}
	void zero()
	{
		for (size_t i = 0; i < size; ++i)
			data[i] = 0;
	}

	float& operator[](int idx)
	{
		assert(idx < (int)size&& idx >= 0);
		return this->data[idx];
	}
	double operator[](int idx) const
	{
		assert(idx < (int)size&& idx >= 0);
		return this->data[idx];
	}

	FeatureVector& operator=(const FeatureVector& rhs);
	FeatureVector& operator=(const std::vector<float> rhs);
	void normalize();
	double distance(const FeatureVector& fv, int type = DISTANCE_TYPE_EUCLIDEAN);

	//TODO this should be protected
	std::vector<float> data;
	size_t size;
};
//直方图设计类
class FeatureHistogram : public FeatureVector
{
public:
	FeatureHistogram();
	~FeatureHistogram();
	FeatureHistogram(int _size, int _label = DEFAULT_LABEL_VALUE)
	{
		resize(_size);
		label = _label;
	}
	FeatureHistogram(const std::vector<float>& _data, int _label = DEFAULT_LABEL_VALUE)
	{
		set(_data);
		label = _label;
	}
	FeatureHistogram(const FeatureHistogram& cpy);

	void setLabel(int _label)
	{
		label = _label;
	}
	int getLabel()
	{
		return label;
	}
	void addAt(int index, float value = 1.0f)
	{
		data[index] += value;
	}

	int label;
};
//分类器基类
class BaseClassifier
{
public:
	BaseClassifier();
	~BaseClassifier();
	BaseClassifier(const BaseClassifier& cpy);
	BaseClassifier& operator=(const BaseClassifier& rhs);
	/*
	virtual bool train() = 0;
	virtual double predict(const FeatureHistogram& hist, bool decisionFunc) = 0;
	virtual void save(const std::string& fileName) = 0;
	virtual void load(const std::string& fileName) = 0;
			*/
	void add(const FeatureHistogram& trainFeature);
	void set(const std::vector<FeatureHistogram>& _trainData);


	std::vector<FeatureHistogram> trainData;
	size_t size;
	size_t length;
	vector<int>svm_labels;
	vector<vector<float>>svm_trainData;
};
//分类器参数类
class SVMParameters
{
public:
	~SVMParameters() {};
	SVMParameters(const SVMParameters& cpy);
	SVMParameters(
		/*int _type = CvSVM::NU_SVC,
		int _kernel = CvSVM::RBF,*/
		double _degree = 3,
		double _gamma = 1,
		double _coef0 = 0.5,
		double _C = 1,
		double _cache = 256,
		double _eps = 0.0001,
		double _nu = 0.5,
		double _p = 0.2,
		int _termType = 1 + 2,
		int _iterations = 1000,
		int _shrinking = 0,
		int _probability = 0,
		int _weight = 0,
		int _kFold = 10);


	SVMParameters& operator=(const SVMParameters& rhs);

	void setDefault();
	void set(/*int _type = CvSVM::NU_SVC,
		int _kernel = CvSVM::RBF,*/
		double _degree = 3,
		double _gamma = 1,
		double _coef0 = 0.5,
		double _C = 1,
		double _cache = 256,
		double _eps = 0.0001,
		double _nu = 0.5,
		double _p = 0.2,
		int _termType = 1 + 2,
		int _iterations = 1000,
		int _shrinking = 0,
		int _probability = 0,
		int _weight = 0,
		int _kFold = 10);

	int type;
	int kernel;
	double degree;
	double gamma;
	double coef0;
	double C;
	double cache;
	double eps;
	double nu;
	double p;
	int termType;
	int iterations;
	int shrinking;
	int probability;
	int weight;
	int kFold;
};
//分类器类
class SVMClassifier : public BaseClassifier
{
public:
	SVMClassifier();
	SVMClassifier(const SVMClassifier& cpy);
	SVMClassifier(const SVMParameters& _params, bool _autoTrain = true);
	~SVMClassifier() {};

	void setAuto(bool _autoTrain) { autoTrain = _autoTrain; }
	bool isAuto() { return autoTrain; }

	void setParameters(const SVMParameters& _params, bool _autoTrain = true);

	bool train();
	double predict(const FeatureHistogram& hist, bool decisionFunc = true);
	void save(const std::string& fileName);
	void load(const std::string& fileName);

private:
	//CvSVM model;
	//CvSVMParams params;
	bool autoTrain;
	int kFold;
};
//快速构建bow算法类
class categorizer
{
private:
	// //从类目名称到数据的map映射
	map<string, Mat> result_objects;
	//存放所有训练图片的BOW
	map<string, Mat> allsamples_bow;
	//从类目名称到训练图集的映射，关键字可以重复出现
	multimap<string, Mat> train_set;
	// 训练得到的SVM
	//Ptr<SVM>* stor_svms;
	//类目名称，每个文件夹的名称，图片的类名
	vector<string> category_name;
	//类目数目，图片的类别的数量
	int categories_size;
	//用SURF特征构造视觉词库的聚类数目
	int clusters;
	//存放训练图片词典
	vector<IPoint>my_vocab_descriptors;

	vector<vector<float>> main_data;
	vector<vector<float>> main_centers;
	//存放每张图的bof
	vector<vector<float>> main_bof;
	//存放索引
	vector<int> main_labels;
	Mat vocab;
	Surf surf;



	//构造训练集合
	void make_train_set();
	// 移除扩展名，用来讲模板组织成类目
	string remove_extention(string);

	Mat getMyMat(Mat);

public:
	int BestID;
	Mat BaseMat;
	//构造函数
	categorizer(int,Mat, multimap<string, Mat> );
	// 聚类得出词典
	void bulid_vacab();
	//构造BOW
	void compute_bow_image();
	//训练分类器
	void trainSvm();
	//将测试图片分类
	void category_By_svm(Mat input_pic);
	Mat Mycluster(const Mat& _descriptors);
	String mysvm(vector<IPoint>& testmat);

	//获取bof函数
	bool getBoF(vector<vector<float>>& input, FeatureHistogram& hist, bool normalized);
	//计算最短的距离
	int mymatch(vector<vector<float>>& input);
	float mydistance(vector<float>& fv, vector<float>& col);
};



//函数声明
double train_data(Surf mysurf, categorizer c);
Mat my_bow(Mat inputmat, Surf mysurf, categorizer c);
Mat ReadFloatImg(const char* szFilename);

Mat ReadFloatImg(const char* szFilename);
std::string base64Decode(const char* Data, int DataByte);
std::string base64Encode(const unsigned char* Data, int DataByte);
std::string Mat2Base64(const cv::Mat& img, std::string imgType);
cv::Mat Base2Mat(std::string& base64_data);





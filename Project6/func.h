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


//��������
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

//ͼ����
class IntegralImg
{
public:
	int Width;		//ͼƬ�Ŀ�
	int Height;		//ͼƬ�ĸ�
	Mat Original;	//ԭʼͼƬ
	Mat Integral;	//����ͼ��
	IntegralImg(Mat img);
	float AreaSum(int x, int y, int dx, int dy);
};

//��Ӧ����
class ResponseLayer
{
public:
	//����ͼ��Ŀ��
	int Width;
	//����ͼ��ĸ߶�
	int Height;
	//ģ�����õĲ���
	int Step;
	//ģ��ĳ��ȵ�1/3
	int Lobe;
	//Lobe*2-1
	int Lobe2;
	//ģ��ĳ���һ�룬�߿�
	int Border;
	//ģ�峤��
	int Size;
	//ģ��Ԫ�ظ���
	int Count;
	//����������
	int Octave;
	//����������
	int Interval;
	//��˹������ͼƬ
	Mat* Data;
	//Laplacian����
	Mat* LapData;

	ResponseLayer(IntegralImg* img, int octave, int interval);
	void BuildLayerData(IntegralImg* img);
	float GetResponse(int x, int y, int step);
	float GetLaplacian(int x, int y, int step);
};

//����Hessian������
class FastHessian
{
public:

	IntegralImg Img;
	//ͼ��ѵ�����
	int Octaves;
	//Ϊͼ�����ÿ���е��м��������ֵ��2����ÿ��ͼ�����������Ĳ���
	int Intervals;
	//Hessian��������ʽ��Ӧֵ����ֵ
	float Threshold;

	map<int, ResponseLayer*> Pyramid;
	//������ʸ������
	vector<IPoint> IPoints;
	//���캯��
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


//surf������
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


//surfʹ����
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
//ֱ��ͼ�����
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
//����������
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
//������������
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
//��������
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
//���ٹ���bow�㷨��
class categorizer
{
private:
	// //����Ŀ���Ƶ����ݵ�mapӳ��
	map<string, Mat> result_objects;
	//�������ѵ��ͼƬ��BOW
	map<string, Mat> allsamples_bow;
	//����Ŀ���Ƶ�ѵ��ͼ����ӳ�䣬�ؼ��ֿ����ظ�����
	multimap<string, Mat> train_set;
	// ѵ���õ���SVM
	//Ptr<SVM>* stor_svms;
	//��Ŀ���ƣ�ÿ���ļ��е����ƣ�ͼƬ������
	vector<string> category_name;
	//��Ŀ��Ŀ��ͼƬ����������
	int categories_size;
	//��SURF���������Ӿ��ʿ�ľ�����Ŀ
	int clusters;
	//���ѵ��ͼƬ�ʵ�
	vector<IPoint>my_vocab_descriptors;

	vector<vector<float>> main_data;
	vector<vector<float>> main_centers;
	//���ÿ��ͼ��bof
	vector<vector<float>> main_bof;
	//�������
	vector<int> main_labels;
	Mat vocab;
	Surf surf;



	//����ѵ������
	void make_train_set();
	// �Ƴ���չ����������ģ����֯����Ŀ
	string remove_extention(string);

	Mat getMyMat(Mat);

public:
	int BestID;
	Mat BaseMat;
	//���캯��
	categorizer(int,Mat, multimap<string, Mat> );
	// ����ó��ʵ�
	void bulid_vacab();
	//����BOW
	void compute_bow_image();
	//ѵ��������
	void trainSvm();
	//������ͼƬ����
	void category_By_svm(Mat input_pic);
	Mat Mycluster(const Mat& _descriptors);
	String mysvm(vector<IPoint>& testmat);

	//��ȡbof����
	bool getBoF(vector<vector<float>>& input, FeatureHistogram& hist, bool normalized);
	//������̵ľ���
	int mymatch(vector<vector<float>>& input);
	float mydistance(vector<float>& fv, vector<float>& col);
};



//��������
double train_data(Surf mysurf, categorizer c);
Mat my_bow(Mat inputmat, Surf mysurf, categorizer c);
Mat ReadFloatImg(const char* szFilename);

Mat ReadFloatImg(const char* szFilename);
std::string base64Decode(const char* Data, int DataByte);
std::string base64Encode(const unsigned char* Data, int DataByte);
std::string Mat2Base64(const cv::Mat& img, std::string imgType);
cv::Mat Base2Mat(std::string& base64_data);





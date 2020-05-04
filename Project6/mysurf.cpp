#include "func.h"
#include <opencv2/opencv.hpp>
#define CV_SVD 1

using namespace cv;
using namespace std;


IntegralImg::IntegralImg(Mat img)
{
	this->Original = img;
	integral(this->Original, this->Integral);
	this->Width = img.cols;
	this->Height = img.rows;
	cout << "this->Width" << endl;//174 ԭͼ160x222 35.5kb
	cout << this->Width << endl;
	cout << "this->Height" << endl;//224
	cout << this->Height << endl;
}
//�������ͼ�� A-B-C+D�������Ͻ���ʼ��ָ���ľ����ڵ������ܺ�����ʹ�С
float IntegralImg::AreaSum(int x, int y, int dx, int dy)
{
	int r1;
	int c1;
	int r2;
	int c2;
	r1 = std::min(x, Height) ;
	c1 = std::min(y, Width) ;
	r2 = std::min(x + dx, Height);
	c2 = std::min(y + dy, Width) ;
	r1 = std::max(r1, 0);
	c1 = std::max(c1, 0);
	r2 = std::max(r2, 0);
	c2 = std::max(c2, 0);
	double A = this->Integral.at<double>(r1, c1);
	double B = this->Integral.at<double>(r2, c1);
	double C = this->Integral.at<double>(r1, c2);
	double D = this->Integral.at<double>(r2, c2);
	return (float)std::max(0.0, A + D - B - C);
}
ResponseLayer::ResponseLayer(IntegralImg* img, int octave, int interval)
{
	this->Step = (int)pow(2.0, octave - 1);
	this->Width = img->Width / this->Step;
	this->Height = img->Height / this->Step;
	this->Lobe = (int)pow(2.0, octave) * interval + 1;
	this->Lobe2 = this->Lobe * 2 - 1;
	this->Size = 3 * this->Lobe;
	this->Border = this->Size / 2;
	this->Count = this->Size * this->Size;
	this->Octave = octave;
	this->Interval = interval;
	this->Data = new Mat(this->Height, this->Width, CV_32FC1);
	this->LapData = new Mat(this->Height, this->Width, CV_32FC1);
	this->BuildLayerData(img);
}

void ResponseLayer::BuildLayerData(IntegralImg* img)
{
	float inverse_area = 1.0 / this->Count;
	float Dxx, Dyy, Dxy;

	for (int r = 0, x = 0; x < Height; r += this->Step, x += 1)
	{
		for (int c = 0, y = 0; y < Width; c += this->Step, y += 1)
		{
			Dxx = img->AreaSum(r - Lobe + 1, c - Border, Lobe2, Size) - img->AreaSum(r - Lobe + 1, c - Lobe / 2, Lobe2, Lobe) * 3;
			Dyy = img->AreaSum(r - Border, c - Lobe + 1, Size, Lobe2) - img->AreaSum(r - Lobe / 2, c - Lobe + 1, Lobe, Lobe2) * 3;
			Dxy = img->AreaSum(r - Lobe, c + 1, Lobe, Lobe) + img->AreaSum(r + 1, c - Lobe, Lobe, Lobe)
				- img->AreaSum(r - Lobe, c - Lobe, Lobe, Lobe) - img->AreaSum(r + 1, c + 1, Lobe, Lobe);
			Dxx *= inverse_area;
			Dyy *= inverse_area;
			Dxy *= inverse_area;

			this->Data->at<float>(x, y) = (Dxx * Dyy - 0.81f * Dxy * Dxy);
			this->LapData->at<float>(x, y) = (Dxx + Dyy >= 0 ? 1 : 0);
		}
	}
}
float ResponseLayer::GetResponse(int x, int y, int step)
{
	int scale = step / this->Step;
	//std::cout<<this->Data->at<float>((x*scale),(y*scale))<<std::endl;
	return this->Data->at<float>((x * scale), (y * scale));
}

float ResponseLayer::GetLaplacian(int x, int y, int step)
{
	int scale = step / this->Step;
	return this->LapData->at<float>((x * scale), (y * scale));
}


//! Gets the distance in descriptor space between Ipoints
float IPoint::operator-(const IPoint& rhs)//���������
{
	float sum = 0.f;
	for (int i = 0; i < 64; ++i)
		sum += (this->descriptor[i] - rhs.descriptor[i]) * (this->descriptor[i] - rhs.descriptor[i]);
	return sqrt(sum);//sqrt��Ǹ�����ƽ����
}



FastHessian::FastHessian(IntegralImg iImg, int octaves, int intervals, float threshold)
	:Octaves(octaves), Intervals(intervals), Img(iImg), Threshold(threshold)
{
	GeneratePyramid();
}
//���ɽ�����
void FastHessian::GeneratePyramid()
{

	for (int o = 1; o <= Octaves; o++)
	{
		for (int i = 1; i <= Intervals; i++)
		{
			int size = 3 * ((int)pow(2.0, o) * i + 1);
			if (!this->Pyramid.count(size))
			{

				this->Pyramid[size] = new ResponseLayer(&Img, o, i);
				//imshow("d",abs((*(Pyramid[size])->Data)*100));
				//cv::waitKey();
			}
		}
	}
}
void FastHessian::GetIPoints()
{
	// Clear the vector of exisiting IPoints
	this->IPoints.clear();

	// Get the response layers
	ResponseLayer* b, * m, * t;
	//��Octaveѭ��
	for (int o = 1; o <= this->Octaves; ++o)
	{
		//һ��Octave������Interval��Size�Ĳ�ֵ
		int step = (int)(3 * pow(2.0, o));
		//���㵱ǰ������Ҫ�����ӵ�Size
		int size = step + 3;
		//�����ӵ����ò���
		int s = (int)pow(2.0, o - 1);
		//�����ͼƬ�Ŀ��
		int width = this->Img.Width / s;
		//�����ͼƬ�ĳ���
		int height = this->Img.Height / s;

		//��Intervalѭ��
		for (int i = 1; i <= this->Intervals - 2; ++i)
		{

			b = this->Pyramid[size];			//��ײ�
			m = this->Pyramid[size + step];		//�м��
			t = this->Pyramid[size + 2 * step];		//��߲�

													//����Border����Border�ڵ����ز���¼Ϊ�ؼ���
													//�����Border������Щ�ɻ�Ϊ��Ҫ����Step��
			int border = (t->Border + 1) / (t->Step);

			//�������еĵ㣬Ѱ�ҷ��ϼ������Ƶĵ�
			//OpenSurf������������а���Border���ڵĵ�
			//����ֱ�Ӻ�����Щ�㣬�ӵ�һ�����������ؿ�ʼ
			for (int r = border + 1; r < height - border; ++r)
			{
				for (int c = border + 1; c < width - border; ++c)
				{
					//�ж��м����м�Ԫ���Ƿ�������Χ26��Ԫ��������
					if (IsExtremum(r, c, s, t, m, b))
					{
						//���������ؼ���Ĳ�ֵ��Ѱ�������ؼ����������
						InterpolateExtremum(r, c, s, t, m, b);
						//cout<<'('<<r<<','<<c<<')'<<endl;
					}
				}
			}
			//����һ����
			size += step;
		}
	}
	//ShowIPoint();
}

//����ֵ����
bool FastHessian::IsExtremum(int r, int c, int step, ResponseLayer* t, ResponseLayer* m, ResponseLayer* b)
{
	// check the candidate point in the middle layer is above thresh 
	float candidate = m->GetResponse(r, c, step);
	if (candidate < this->Threshold)
		return 0;

	for (int rr = -1; rr <= 1; ++rr)
	{
		for (int cc = -1; cc <= 1; ++cc)
		{
			// if any response in 3x3x3 is greater candidate not maximum
			if (
				t->GetResponse(r + rr, c + cc, step) >= candidate ||							//�붥��9��Ԫ�رȽ�
				((rr != 0 || cc != 0) && m->GetResponse(r + rr, c + cc, step) >= candidate) ||	//���м��8��Ԫ�رȽ�
				b->GetResponse(r + rr, c + cc, step) >= candidate								//��ײ�9��Ԫ�رȽ�
				)
				return 0;
		}
	}
	return 1;
}
//���������
void FastHessian::InterpolateExtremum(int r, int c, int step, ResponseLayer* t, ResponseLayer* m, ResponseLayer* b)
{
	// get the step distance between filters
	// check the middle filter is mid way between top and bottom
	int filterStep = (m->Size - b->Size);
	assert(filterStep > 0 && t->Size - m->Size == m->Size - b->Size);

	// Get the offsets to the actual location of the extremum
	double xi = 0, xr = 0, xc = 0;
	InterpolateStep(r, c, step, t, m, b, &xi, &xr, &xc);

	// If point is sufficiently close to the actual extremum
	if (fabs(xi) < 0.5f && fabs(xr) < 0.5f && fabs(xc) < 0.5f)
	{
		IPoint p;
		p.x = static_cast<float>((c + xc) * step);
		p.y = static_cast<float>((r + xr) * step);
		p.scale = static_cast<float>((0.1333f) * (m->Size + xi * filterStep));
		p.laplacian = static_cast<int>(m->GetLaplacian(r, c, step));
		this->IPoints.push_back(p);
	}
}
//��̩��չ����⼫ֵ��
void FastHessian::InterpolateStep(int r, int c, int step, ResponseLayer* t, ResponseLayer* m, ResponseLayer* b,
	double* xi, double* xr, double* xc)
{
	Mat dD, H, H_inv, X;

	dD = Deriv3D(r, c, step, t, m, b);
	//cout<<dD<<endl;
	H = Hessian3D(r, c, step, t, m, b);
	//cout<<H<<endl;
	invert(H, H_inv, CV_SVD);
	//cout<<H_inv<<endl;
	gemm(H_inv, dD, -1, NULL, 0, X, 0);

	*xc = X.at<double>(0, 0);
	*xr = X.at<double>(1, 0);
	*xi = X.at<double>(2, 0);
}
//����һ�׵���
Mat FastHessian::Deriv3D(int r, int c, int step, ResponseLayer* t, ResponseLayer* m, ResponseLayer* b)
{
	double dx, dy, ds;
	dx = (m->GetResponse(r, c + 1, step) - m->GetResponse(r, c - 1, step)) / 2.0;
	dy = (m->GetResponse(r + 1, c, step) - m->GetResponse(r - 1, c, step)) / 2.0;
	ds = (t->GetResponse(r, c, step) - b->GetResponse(r, c, step)) / 2.0;

	//����һ�׵���
	Mat dI = (Mat_<double>(3, 1) << dx, dy, ds);

	return dI;
}

//������׵���
Mat FastHessian::Hessian3D(int r, int c, int step, ResponseLayer* t, ResponseLayer* m, ResponseLayer* b)
{
	double v, dxx, dyy, dss, dxy, dxs, dys;

	v = m->GetResponse(r, c, step);
	dxx = m->GetResponse(r, c + 1, step) + m->GetResponse(r, c - 1, step) - 2 * v;
	dyy = m->GetResponse(r + 1, c, step) + m->GetResponse(r - 1, c, step) - 2 * v;
	dss = t->GetResponse(r, c, step) + b->GetResponse(r, c, step) - 2 * v;
	dxy = (m->GetResponse(r + 1, c + 1, step) - m->GetResponse(r + 1, c - 1, step) -
		m->GetResponse(r - 1, c + 1, step) + m->GetResponse(r - 1, c - 1, step)) / 4.0;
	dxs = (t->GetResponse(r, c + 1, step) - t->GetResponse(r, c - 1, step) -
		b->GetResponse(r, c + 1, step) + b->GetResponse(r, c - 1, step)) / 4.0;
	dys = (t->GetResponse(r + 1, c, step) - t->GetResponse(r - 1, c, step) -
		b->GetResponse(r + 1, c, step) + b->GetResponse(r - 1, c, step)) / 4.0;

	//����Hessian����
	Mat H = (Mat_<double>(3, 3) <<
		dxx, dxy, dxs,
		dxy, dyy, dys,
		dxs, dys, dss);

	return H;
}

//! SURF priors (these need not be done at runtime)
const float pi = 3.14159f;

//! lookup table for 2d gaussian (sigma = 2.5) where (0,0) is top left and (6,6) is bottom right
const float gauss25[7][7] = {
	0.02546481,	0.02350698,	0.01849125,	0.01239505,	0.00708017,	0.00344629,	0.00142946,
	0.02350698,	0.02169968,	0.01706957,	0.01144208,	0.00653582,	0.00318132,	0.00131956,
	0.01849125,	0.01706957,	0.01342740,	0.00900066,	0.00514126,	0.00250252,	0.00103800,
	0.01239505,	0.01144208,	0.00900066,	0.00603332,	0.00344629,	0.00167749,	0.00069579,
	0.00708017,	0.00653582,	0.00514126,	0.00344629,	0.00196855,	0.00095820,	0.00039744,
	0.00344629,	0.00318132,	0.00250252,	0.00167749,	0.00095820,	0.00046640,	0.00019346,
	0.00142946,	0.00131956,	0.00103800,	0.00069579,	0.00039744,	0.00019346,	0.00008024
};

//-------------------------------------------------------

SurfDescriptor::SurfDescriptor(IntegralImg& img, std::vector<IPoint>& iPoints) :Img(img), IPoints(iPoints)
{

}




//����Ҫ��ȡ������ת�����Ե�����

//��ȡ��ǰ�ؼ����ڸ��������������
void SurfDescriptor::GetOrientation()
{
	for (int i = 0; i < this->IPoints.size(); i++)
	{
		const int pCount = 109;
		IPoint& p = IPoints[i];
		float gauss = 0.f;
		int s = fRound(p.scale), r = fRound(p.y), c = fRound(p.x);
		float resX[pCount], resY[pCount], Ang[pCount];
		int id[] = { 6,5,4,3,2,1,0,1,2,3,4,5,6 };

		int idx = 0;

		//����6��scale�������haar����
		for (int i = -6; i <= 6; i++)
		{
			for (int j = -6; j <= 6; j++)
			{
				if (i * i + j * j < 36)
				{
					//��4��scale��haar������ȡx y�����ϵ��ݶ�����
					//Ϊ����4��sigma��
					gauss = gauss25[id[i + 6]][id[j + 6]];
					resX[idx] = gauss * haarX(r + j * s, c + i * s, 4 * s);
					resY[idx] = gauss * haarY(r + j * s, c + i * s, 4 * s);
					//���㵱ǰ��ķ�������
					Ang[idx] = getAngle(resX[idx], resY[idx]);
					idx++;
				}
			}
		}

		//����������
		float sumX = 0.f, sumY = 0.f;
		float maxX = 0.f, maxY = 0.f;
		float max = 0.f, orientation = 0.f;
		float ang1 = 0.f, ang2 = 0.f;

		//����pi/3���ε�������
		//����Ϊ0.15
		float pi3 = pi / 3.0f;
		for (ang1 = 0; ang1 < 2 * pi; ang1 += 0.15f)
		{
			ang2 = (ang1 + pi3 > 2 * pi ? ang1 - 5.0f * pi3 : ang1 + pi3);
			sumX = sumY = 0.f;
			for (int k = 0; k < pCount; k++)
			{
				const float& ang = Ang[k];
				if (ang1 < ang2 && ang1 < ang && ang < ang2)
				{
					sumX += resX[k];
					sumY += resY[k];
				}
				//��Ȼ��or��������һ������
				else if (ang1 > ang2 &&
					((0 < ang && ang < ang2) || (ang1 < ang && ang < 2 * pi)))
				{
					sumX += resX[k];
					sumY += resY[k];
				}
			}

			//�ҵ�������Ҳ����ģ���ķ���
			if (sumX * sumX + sumY * sumY > max)
			{
				max = sumX * sumX + sumY * sumY;
				maxX = sumX;
				maxY = sumY;
			}
		}

		p.orientation = getAngle(maxX, maxY);
	}
}

//��������������
void SurfDescriptor::DrawOrientation()
{
	int r1, c1, c2, r2;
	for (int i = 0; i < this->IPoints.size(); i++)
	{
		r1 = fRound(IPoints[i].y);
		c1 = fRound(IPoints[i].x);
		c2 = fRound(10 * cos(IPoints[i].orientation)) + c1;
		r2 = fRound(10 * sin(IPoints[i].orientation)) + r1;
		cv::line(this->Img.Original, cv::Point(c1, r1), cv::Point(c2, r2), cv::Scalar(0, 255, 0));
	}
	imshow("d", this->Img.Original);
}

//�����������ֵ����ȡ4*4*4=64ά��Haar����
void SurfDescriptor::GetDescriptor()
{
	//OpenSURF ���������д�����˺������
	//������Ϊ��Ч�ʣ�����д�úܾ���
	//���ڽ������㸽�����򻮷ֳ�4*4��������
	//��ô��������ĳ������������ĵ�
	int o[] = { -7, -2, 3, 8 };
	//int so[]={-2, -1, 0, 1, 2};

	for (int t = 0; t < this->IPoints.size(); t++)
	{
		IPoint& p = IPoints[t];
		float scale = p.scale;
		float* desp = p.descriptor;
		int x = fRound(p.x);
		int y = fRound(p.y);
		float co = cos(p.orientation);
		float si = sin(p.orientation);
		float cx = -0.5f, cy = 0.f; //Subregion centers for the 4x4 gaussian weighting
		int count = 0;
		float len = 0.f;

		for (int i = 0; i < 4; i++)
		{
			cx += 1.f;
			cy = -0.5f;
			for (int j = 0; j < 4; j++)
			{
				int xs = fRound(RotateX(scale * o[i], scale * o[j], si, co) + x);
				int ys = fRound(RotateY(scale * o[i], scale * o[j], si, co) + y);
				float dx = 0.f, dy = 0.f, mdx = 0.f, mdy = 0.f;

				cy += 1.f;
				for (int k = o[i] - 5; k <= o[i] + 3; k++)
				{
					for (int l = o[j] - 5; l <= o[j] + 3; l++)
					{
						int sample_x = fRound(RotateX(scale * k, scale * l, si, co) + x);
						int sample_y = fRound(RotateY(scale * k, scale * l, si, co) + y);

						//Ϊ����2.5*scale����������д����3.3*scale
						float gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.5f * scale);
						float rx = haarX(sample_y, sample_x, 2 * fRound(scale));
						float ry = haarY(sample_y, sample_x, 2 * fRound(scale));

						float rrx = gauss_s1 * RotateX(rx, ry, si, co);
						float rry = gauss_s1 * RotateY(rx, ry, si, co);

						dx += rrx;
						dy += rry;
						mdx += fabs(rrx);
						mdy += fabs(rry);
					}
				}

				float gauss_s2 = gaussian(cx - 2.f, cy - 2.f, 1.5f);

				desp[count++] = dx * gauss_s2;
				desp[count++] = dy * gauss_s2;
				desp[count++] = mdx * gauss_s2;
				desp[count++] = mdy * gauss_s2;

				len += (dx * dx + dy * dy + mdx * mdx + mdy * mdy) * gauss_s2 * gauss_s2;
			}
		}

		len = sqrt(len);
		for (int i = 0; i < 64; ++i)
			desp[i] /= len;

	}

}

//����Ϊ��������
//���ݽǶȣ���ת����
inline float SurfDescriptor::RotateX(float x, float y, float si, float co)
{
	return -x * si + y * co;
}

inline float SurfDescriptor::RotateY(float x, float y, float si, float co)
{
	return x * co + y * si;
}

//! Round float to nearest integer
inline int SurfDescriptor::fRound(float flt)
{
	return (int)floor(flt + 0.5f);
}
//-------------------------------------------------------

//! Calculate the value of the 2d gaussian at x,y
inline float SurfDescriptor::gaussian(int x, int y, float sig)
{
	return (1.0f / (2.0f * pi * sig * sig)) * exp(-(x * x + y * y) / (2.0f * sig * sig));
}

//-------------------------------------------------------

//! Calculate the value of the 2d gaussian at x,y
inline float SurfDescriptor::gaussian(float x, float y, float sig)
{
	return 1.0f / (2.0f * pi * sig * sig) * exp(-(x * x + y * y) / (2.0f * sig * sig));
}

//-------------------------------------------------------

//! Calculate Haar wavelet responses in x direction
inline float SurfDescriptor::haarX(int row, int column, int s)
{
	return Img.AreaSum(row - s / 2, column, s, s / 2)
		- Img.AreaSum(row - s / 2, column - s / 2, s, s / 2);
}

//-------------------------------------------------------

//! Calculate Haar wavelet responses in y direction
inline float SurfDescriptor::haarY(int row, int column, int s)
{
	return Img.AreaSum(row, column - s / 2, s / 2, s)
		- Img.AreaSum(row - s / 2, column - s / 2, s / 2, s);
}

//-------------------------------------------------------

//! Get the angle from the +ve x-axis of the vector given by (X Y)
float SurfDescriptor::getAngle(float X, float Y)
{
	if (X > 0 && Y >= 0)
		return atan(Y / X);

	if (X < 0 && Y >= 0)
		return pi - atan(-Y / X);

	if (X < 0 && Y < 0)
		return pi + atan(Y / X);

	if (X > 0 && Y < 0)
		return 2 * pi - atan(-Y / X);

	return 0;
}


vector<IPoint> Surf::GetAllFeatures(Mat img)
{
	//��ʼ��ͼ�����������ԭʼͼ�񣬻���ͼ�񣬸ߣ���
	IntegralImg IImg(img);
	FastHessian fh(IImg, 4, 4, 0.0001);

	fh.GetIPoints();
	SurfDescriptor sd(IImg, fh.IPoints);
	sd.GetOrientation();
	sd.GetDescriptor();

	//clock_t start;
	//clock_t end;
	//start = clock();
	//IntegralImg IImg(img);
	//end = clock();
	//cout << "IntegralImg took: " << float(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
	//start = clock();
	//FastHessian fh(IImg, 4, 4, 0.0001);
	//fh.GetIPoints();
	//end = clock();
	//std::cout << "FastHessian took: " << float(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
	//start = clock();
	//SurfDescriptor sd(IImg, fh.IPoints);
	//sd.GetOrientation();
	//sd.GetDescriptor();
	//end = clock();
	//std::cout << "Descriptor took: " << float(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
	return fh.IPoints;
}
Mat ReadFloatImg(const char* szFilename)
{
	Mat iImg = imread(szFilename, 0);
	Mat fImg;
	iImg.convertTo(fImg, CV_32FC1);
	fImg /= 255.0;
	return fImg;
}
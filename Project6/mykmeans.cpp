#include <iostream>
#include "mykmeans.h"
#include <algorithm>
#include <limits>
#include <fstream>
#include <string>
#include <random>
#include <vector>
#include <typeinfo>
#include <opencv2/opencv.hpp>

namespace MYKMEANS {

	namespace {

		template<typename T>
		void generate_random_center(const std::vector<std::vector<T>>& box, std::vector<T>& center)
		{
			std::random_device rd;
			//���������
			std::mt19937 generator(rd());
			//Ĭ�Ϸ��� double �͸���ֵ�������ֲ�
			std::uniform_real_distribution<T> distribution((T)0, (T)0.0001);

			int dims = box.size();
			T margin = 1.f / dims;
			//ѭ��64
			for (int j = 0; j < dims; j++) {
				//����[0.0001--1]�����Էֲ�
				center[j] = (distribution(generator) * (1. + margin * 2.) - margin) * (box[j][1] - box[j][0]) + box[j][0];
			}
		}

		//ʹ������ƽ����(WCSS within-cluster  sum of squares)��С
		template<typename T>
		inline T norm_L2_Sqr(const T* a, const T* b, int n)
		{
			double s = 0.f;
			for (int i = 0; i < n; i++) {
				double v = double(a[i] - b[i]);
				s += v * v;
			}
			return s;
		}

		template<typename T>
		void distance_computer(std::vector<double>& distances, std::vector<int>& labels, const std::vector<std::vector<T>>& data,
			const std::vector<std::vector<T>>& centers, bool only_distance = false)
		{
			const int K = centers.size();
			const int dims = centers[0].size();
			//ѭ���㼯����
			for (int i = 0; i < distances.size(); ++i) {
				const std::vector<T> sample = data[i];

				if (only_distance) {
					const std::vector<T> center = centers[labels[i]];
					distances[i] = norm_L2_Sqr(sample.data(), center.data(), dims);
					continue;
				}

				int k_best = 0;
				double min_dist = std::numeric_limits<double>::max(); // DBL_MAX
				  //��ѭ���Ľ������õ�����������������ľ������ģ��þ��������Ǳ���ѭ���������ɵģ�
				for (int k = 0; k < K; ++k) {
					const std::vector<T> center = centers[k];
					const double dist = norm_L2_Sqr(sample.data(), center.data(), dims);//���������������ĳ��center�ľ���

					if (min_dist > dist) {
						min_dist = dist;
						k_best = k;
					}
				}

				distances[i] = min_dist;
				labels[i] = k_best;
			}
		}

	} // namespace

	template<typename T>
	int kmeans(const std::vector<std::vector<T>>& data, int K, std::vector<int>& best_labels,
		std::vector<std::vector<T>>& centers, double& compactness_measure,
		int max_iter_count, double epsilon, int attempts, int flags)
	{
		CHECK(flags == KMEANS_RANDOM_CENTERS);
		//��ȡ������󳤶�
		int N = data.size();
		//���󳤶���Ҫ���ָ������ʱ����Ϊ���ࣻ
		CHECK(K > 0 && N >= K);

		//ָ��һ�е�����
		int dims = data[0].size();
		//ָ��K-Means�㷨ִ�еĴ�����ÿ���㷨ִ�еĽ���ǲ�һ���ģ�ѡ����õ��Ǵν�����
		attempts = std::max(attempts, 1);
		//�����������ĳ���
		best_labels.resize(N);
		//�����м����label
		std::vector<int> labels(N);

		//������յľ�ֵ��ľ���
		centers.resize(K);
		std::vector<std::vector<T>> centers_(K), old_centers(K);
		//��ʼ��tempΪ�������ĵ�һ��������С��ȫΪ0.0
		std::vector<T> temp(dims, (T)0.);
		//��ʼ����������Ϊ������+��һ��������С�ľ���
		for (int i = 0; i < K; ++i) {
			centers[i].resize(dims);
			centers_[i].resize(dims);
			old_centers[i].resize(dims);
		}

		//���ֵ
		compactness_measure = std::numeric_limits<double>::max(); // DBL_MAX
		//��ʼ��Ϊ0.0
		double compactness = 0.;

		//0.001epsilon
		epsilon = std::max(epsilon, (double)0.);
		epsilon *= epsilon;//

		//�޶������Ĵ�����2��100��
		max_iter_count = std::min(std::max(max_iter_count, 2), 100);

		//������ֳ�һ���ʱ�򣬽�kmeans�㷨��ִ��һ�Σ�����������������
		if (K == 1) {
			attempts = 1;
			max_iter_count = 2;
		}

		//�½�box��������64����ʼ����vector
		std::vector<std::vector<T>> box(dims);
		for (int i = 0; i < dims; ++i) {
			box[i].resize(2);
		}

		//��ʼ��������ĸ���dists�������counters
		std::vector<double> dists(N, 0.);
		std::vector<int> counters(K);

		//box�е�����ָ�붼����㼯��ָ��ͷ
		const T* sample = data[0].data();
		//��box�зŽ�64*2�����ݲ���
		for (int i = 0; i < dims; ++i) {
			box[i][0] = sample[i];
			box[i][1] = sample[i];
		}

		//�����һ�н��бȽϣ����㼯��ÿһ�ж����һ�����Ƚϣ��õ��㼯��ÿһ�е����ֵ����Сֵ
		for (int i = 1; i < N; ++i) {
			sample = data[i].data();
			//�Ƚ�ÿһ�е���Сֵ�����ֵ
			for (int j = 0; j < dims; ++j) {
				T v = sample[j];
				box[j][0] = std::min(box[j][0], v);
				box[j][1] = std::max(box[j][1], v);
			}
		}

		//ѭ��100��
		for (int a = 0; a < attempts; ++a) {
			double max_center_shift = std::numeric_limits<double>::max(); // DBL_MAX���ֵ
			//һֱѭ���𣬵���һ�ν�ѭ����ʱ��centers_���и�ֵ��K������������ڵڶ��ν����ʱ�򽻻���old_centers���ٴ�Ϊ��
			for (int iter = 0;;) {
				//��ȫ����
				centers_.swap(old_centers);

				if (iter == 0 && (a > 0 || true)) {
					for (int k = 0; k < K; ++k) {
						//�����[0.0001-1]��K�������
						generate_random_center(box, centers_[k]);
					}
				}
				else {
					// compute centers�ڶ��ν�����ʱ��centers_Ϊ�գ�ȫ����ʼ��Ϊ0���������
					for (auto& center : centers_) {
						std::for_each(center.begin(), center.end(), [](T& v) {v = (T)0; });
					}
					//��K����ʼ��
					std::for_each(counters.begin(), counters.end(), [](int& v) {v = 0; });
					//ѭ���㼯ÿһ�У�//��ÿ�������������з���center�еõ�ÿ��
					for (int i = 0; i < N; ++i) {
						sample = data[i].data();//�õ�����������
						//�õ������������������� ������
						//labels����ʲôʱ���ʼ���ģ�labels�ڵ�һ��ѭ��ʱ���и�ֵ�����˹�ָ����֮����ÿ��ѭ���н��и�ֵ
						int k = labels[i];
						//�õ�����������������������
						auto& center = centers_[k];

						//center�е�ֵ��ʼΪ��
						//forѭ���������ǽ����������������ۼӵ�center�У����������� +4 ����������ѭ������
						int j;
						for (j = 0; j <= dims-4; j += 4) {
							//center[j] += sample[j];
							float t0 = center[j] + sample[j];
							float t1 = center[j + 1] + sample[j + 1];
							center[j] = t0;
							center[j + 1] = t1;
							t0 = center[j + 2] + sample[j + 2];
							t1 = center[j + 3] + sample[j + 3];
							center[j + 2] = t0;
							center[j + 3] = t1;
						}
						//����һ��forѭ����ʣ��һ��β���ۼӽ�ȥ
						for (; j < dims; j++)
							center[j] += sample[j];

						counters[k]++;
					}

					if (iter > 0) max_center_shift = 0;

					for (int k = 0; k < K; ++k) {
						if (counters[k] != 0) continue;
						/*���ĳ���ؿ���ȥ�ǿյģ���ô
							1.�ҵ����ļ�Ⱥ
							2.������Ⱥ�����ҵ������ĵ���Զ��λ��
							3.�����Ⱥ�����ų���Զ�ĵ㣬���γ�һ���µ�1��Ⱥ��*/
						int max_k = 0;
						//�����ǵ�1��ѭ��������ʱ������old center,�ܹ���������ѭ��������ƫ����
						for (int k1 = 1; k1 < K; ++k1) {
							if (counters[max_k] < counters[k1])
								max_k = k1;
						}

						double max_dist = 0;
						int farthest_i = -1;
						auto& new_center = centers_[k];
						auto& old_center = centers_[max_k];
						auto& _old_center = temp; // normalized
						T scale = (T)1.f / counters[max_k];
						//�Ե�k��centerȡ���������������ľ�ֵ
						for (int j = 0; j < dims; j++) {
							_old_center[j] = old_center[j] * scale;
						}

						for (int i = 0; i < N; ++i) {
							if (labels[i] != max_k)
								continue;
							sample = data[i].data();
							double dist = norm_L2_Sqr(sample, _old_center.data(), dims);
							//��������ÿ��������ľ��벢���б��
							if (max_dist <= dist) {
								max_dist = dist;
								farthest_i = i;
							}
						}

						counters[max_k]--;
						counters[k]++;
						labels[farthest_i] = k;
						sample = data[farthest_i].data();

						for (int j = 0; j < dims; ++j) {
							old_center[j] -= sample[j];
							new_center[j] += sample[j];
						}
					}

					for (int k = 0; k < K; ++k) {
						auto& center = centers_[k];
						CHECK(counters[k] != 0);
						//�Ե�k��centerȡ���������������ľ�ֵ
						T scale = (T)1.f / counters[k];
						for (int j = 0; j < dims; ++j) {
							center[j] *= scale;
						}

						if (iter > 0) {
							double dist = 0;
							const auto old_center = old_centers[k];
							for (int j = 0; j < dims; j++) {
								T t = center[j] - old_center[j];
								dist += t * t;
							}
							//��ʱ���������ƫ���������С���趨ֵ������ѭ��
							max_center_shift = std::max(max_center_shift, dist);
						}
					}
				}

				bool isLastIter = (++iter == std::max(max_iter_count, 2) || max_center_shift <= epsilon);

				// assign labels  ���·���labels
				std::for_each(dists.begin(), dists.end(), [](double& v) {v = 0; });

				distance_computer(dists, labels, data, centers_, isLastIter);
				std::for_each(dists.cbegin(), dists.cend(), [&compactness](double v) { compactness += v; });

				if (isLastIter) break;
			}

			if (compactness < compactness_measure) {
				compactness_measure = compactness;
				for (int i = 0; i < K; ++i) {
					memcpy(centers[i].data(), centers_[i].data(), sizeof(T) * dims);
				}
				memcpy(best_labels.data(), labels.data(), sizeof(int) * N);
			}
		}

		return 0;
	}

	template int kmeans<float>(const std::vector<std::vector<float>>&, int K, std::vector<int>&, std::vector<std::vector<float>>&, double&,
		int max_iter_count, double epsilon, int attempts, int flags);
	template int kmeans<double>(const std::vector<std::vector<double>>&, int K, std::vector<int>&, std::vector<std::vector<double>>&, double&,
		int max_iter_count, double epsilon, int attempts, int flags);

}

//
int save_images(const std::vector<cv::Mat>& src, const std::string& name, int row_image_count)
{
	int rows = ((src.size() + row_image_count - 1) / row_image_count);
	int width = src[0].cols, height = src[0].rows;
	cv::Mat dst(height * rows, width * row_image_count, CV_8UC1);

	for (int i = 0; i < src.size(); ++i) {
		int row_start = (i / row_image_count) * height;
		int row_end = row_start + height;
		int col_start = i % row_image_count * width;
		int col_end = col_start + width;
		cv::Mat part = dst(cv::Range(row_start, row_end), cv::Range(col_start, col_end));
		src[i].copyTo(part);
	}

	cv::imwrite(name, dst);

	return 0;
}

int mat_horizontal_concatenate()
{
#ifdef _MSC_VER
	const std::string path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_0_and_1/" };
#else
	const std::string path{ "data/images/digit/handwriting_0_and_1/" };
#endif

	std::vector<std::string> prefix{ "0_", "1_", "2_", "3_" };
	const int every_class_number{ 20 };
	const int category_number{ (int)prefix.size() };
	std::vector<std::vector<cv::Mat>> mats(category_number);

	cv::Mat mat = cv::imread(path + "0_1.jpg", 0);
	CHECK(!mat.empty());

	const int width{ mat.cols }, height{ mat.rows };

	int count{ 0 };
	for (const auto& value : prefix) {
		for (int i = 1; i <= every_class_number; ++i) {
			std::string name = path + value + std::to_string(i) + ".jpg";
			cv::Mat mat = cv::imread(name, 0);
			if (mat.empty()) {
				fprintf(stderr, "read image fail: %s\n", name.c_str());
				return -1;
			}
			if (width != mat.cols || height != mat.rows) {
				fprintf(stderr, "image size not equal\n");
				return -1;
			}

			mats[count].push_back(mat);
		}

		++count;
	}

	std::vector<cv::Mat> middle(category_number);
	for (int i = 0; i < category_number; ++i) {
		cv::hconcat(mats[i].data(), mats[i].size(), middle[i]);
	}

	cv::Mat dst;
	cv::vconcat(middle.data(), middle.size(), dst);
#ifdef _MSC_VER
	cv::imwrite("E:/GitCode/NN_Test/data/result.jpg", dst);
#else
	cv::imwrite("data/result.jpg", dst);
#endif

	return 0;
}

int compare_file(const std::string& name1, const std::string& name2)
{
	std::ifstream infile1;
	infile1.open(name1.c_str(), std::ios::in | std::ios::binary);
	if (!infile1.is_open()) {
		fprintf(stderr, "failed to open file\n");
		return -1;
	}

	std::ifstream infile2;
	infile2.open(name2.c_str(), std::ios::in | std::ios::binary);
	if (!infile2.is_open()) {
		fprintf(stderr, "failed to open file\n");
		return -1;
	}

	size_t length1 = 0, length2 = 0;

	infile1.read((char*)&length1, sizeof(size_t));
	infile2.read((char*)&length2, sizeof(size_t));

	if (length1 != length2) {
		fprintf(stderr, "their length is mismatch: required length: %d, actual length: %d\n", length1, length2);
		return -1;
	}

	double* data1 = new double[length1];
	double* data2 = new double[length2];

	for (int i = 0; i < length1; i++) {
		infile1.read((char*)&data1[i], sizeof(double));
		infile2.read((char*)&data2[i], sizeof(double));

		if (data1[i] != data2[i]) {
			fprintf(stderr, "no equal: %d: %f, %f\n", i, data1[i], data2[i]);
		}
	}

	delete[] data1;
	delete[] data2;

	infile1.close();
	infile2.close();
}

template<typename T>
void generator_real_random_number(T* data, int length, T a, T b)
{
	//std::random_device rd; std::mt19937 generator(rd()); // ÿ�β������̶��Ĳ�ͬ��ֵ
	std::default_random_engine generator; // ÿ�β����̶��Ĳ�ͬ��ֵ
	std::uniform_real_distribution<T> distribution(a, b);

	for (int i = 0; i < length; ++i) {
		data[i] = distribution(generator);
	}
}

template<typename T>
int read_txt_file(const char* name, std::vector<std::vector<T>>& data, const char separator, const int rows, const int cols)
{
	if (typeid(float).name() != typeid(T).name()) {
		fprintf(stderr, "string convert to number only support float type\n");
		return -1;
	}

	std::ifstream fin(name, std::ios::in);
	if (!fin.is_open()) {
		fprintf(stderr, "open file fail: %s\n", name);
		return -1;
	}

	std::string line, cell;
	int col_count = 0, row_count = 0;
	data.clear();

	while (std::getline(fin, line)) {
		col_count = 0;
		++row_count;
		std::stringstream line_stream(line);
		std::vector<T> vec;

		while (std::getline(line_stream, cell, separator)) {
			++col_count;
			vec.emplace_back(std::stof(cell));
		}

		CHECK(cols == col_count);
		data.emplace_back(vec);
	}

	CHECK(rows == row_count);

	fin.close();
	return 0;
}

template void generator_real_random_number<float>(float*, int, float, float);
template void generator_real_random_number<double>(double*, int, double, double);
//template int read_txt_file<int>(const char*, std::vector<std::vector<int>>&, const char, const int, const int);
template int read_txt_file<float>(const char*, std::vector<std::vector<float>>&, const char, const int, const int);
//template int read_txt_file<double>(const char*, std::vector<std::vector<double>>&, const char, const int, const int);



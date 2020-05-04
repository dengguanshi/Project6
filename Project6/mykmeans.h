#pragma once
#ifndef FBC_SRC_NN_KMEANS_HPP_
#define FBC_SRC_NN_KMEANS_HPP_
#include <vector>

namespace MYKMEANS {
	typedef enum KmeansFlags {
		//每次尝试随机选择初始中心
		KMEANS_RANDOM_CENTERS = 0,
		// Use kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007]
		//KMEANS_PP_CENTERS = 2,
		// During the first (and possibly the only) attempt, use the
		//user-supplied labels instead of computing them from the initial centers. For the second and
		//further attempts, use the random or semi-random centers. Use one of KMEANS_\*_CENTERS flag
		//to specify the exact method.
		//KMEANS_USE_INITIAL_LABELS = 1
	} KmeansFlags;
	template<typename T>
	int kmeans(const std::vector<std::vector<T>>& data, int K, std::vector<int>& best_labels, std::vector<std::vector<T>>& centers, double& compactness_measure,
		int max_iter_count = 100, double epsilon = 0.001, int attempts = 3, int flags = KMEANS_RANDOM_CENTERS);
} // namespace ANN

#endif // FBC_SRC_NN_KMEANS_HPP_
#pragma once
#ifndef FBC_NN_COMMON_HPP_
#define FBC_NN_COMMON_HPP_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#define PI 3.14159265358979323846

#define CHECK(x) { \
	if (x) {} \
	else { fprintf(stderr, "Check Failed: %s, file: %s, line: %d\n", #x, __FILE__, __LINE__); return -1; } \
}

template<typename T>
void generator_real_random_number(T* data, int length, T a = (T)0, T b = (T)1);
int mat_horizontal_concatenate();
int save_images(const std::vector<cv::Mat>& src, const std::string& name, int row_image_count);
template<typename T>
int read_txt_file(const char* name, std::vector<std::vector<T>>& data, const char separator, const int rows, const int cols);
#endif // FBC_NN_COMMON_HPP_

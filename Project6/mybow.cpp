#include "func.h"


using namespace std;
#define TRAIN_FOLDER "C:/Java/MyBOW/data/train_images/"
#define DATA_FOLDER "C:/Java/MyBOW/data/"



// 移除扩展名，用来讲模板组织成类目
string categorizer::remove_extention(string full_name)
{
    //find_last_of找出字符最后一次出现的地方
    int last_index = full_name.find_last_of(".");
    string name = full_name.substr(0, last_index);
    return name;
}

// 构造函数
categorizer::categorizer(int _clusters,Mat baseMat, multimap<string, Mat> train_label_mat)
{
    cout << "执行categorizer构造函数..." << endl;
    clusters = _clusters;
    //初始化指针
    int minHessian = 400;
    BaseMat = baseMat;
    //当使用mongdb里的数据时，打开这行代码
    ifstream fin("vocabulary.txt");
    //如果之前已经生成好，就不需要重新聚类生成词典
    if (fin)
    {
        cout << "跳过赋值train_set" << endl;
    }
    else {
        train_set = train_label_mat;//!!!
    }

    //读取训练集
    make_train_set();
}

//构造训练集合
void categorizer::make_train_set()
{
    cout << "读取训练集..." << endl;

    //Mat temp = imread("C:\\Java\\MyBOW\\data\\train_images\\1\\1_1.jpg", 0);
    //pair<string, Mat> p0("1", temp);
    //Mat temp1 = imread("C:\\Java\\MyBOW\\data\\train_images\\1\\1_2.jpg", 0);
    //pair<string, Mat> p1("1", temp1);
    //Mat temp2 = imread("C:\\Java\\MyBOW\\data\\train_images\\2\\2_1.jpg", 0);
    //pair<string, Mat> p2("2", temp2);
    //Mat temp3 = imread("C:\\Java\\MyBOW\\data\\train_images\\2\\2_2.jpg", 0);
    //pair<string, Mat> p3("2", temp3);
    //Mat temp4 = imread("C:\\Java\\MyBOW\\data\\train_images\\3\\3_1.jpg", 0);
    //pair<string, Mat> p4("3", temp4);
    //Mat temp5 = imread("C:\\Java\\MyBOW\\data\\train_images\\3\\3_2.jpg", 0);
    //pair<string, Mat> p5("3", temp5);
    //Mat temp6 = imread("C:\\Java\\MyBOW\\data\\train_images\\4\\4_1.jpg", 0);
    //pair<string, Mat> p6("4", temp6);
    //Mat temp7 = imread("C:\\Java\\MyBOW\\data\\train_images\\4\\4_2.jpg", 0);
    //pair<string, Mat> p7("4", temp7);
    //Mat temp8 = imread("C:\\Java\\MyBOW\\data\\train_images\\5\\5_1.jpg", 0);
    //pair<string, Mat> p8("5", temp8);
    //Mat temp9 = imread("C:\\Java\\MyBOW\\data\\train_images\\6\\6_1.jpg", 0);
    //pair<string, Mat> p9("6", temp9);
    //Mat temp10 = imread("C:\\Java\\MyBOW\\data\\train_images\\6\\6_2.jpg", 0);
    //pair<string, Mat> p10("6", temp10);
    //train_set.insert(p0);
    //train_set.insert(p1);
    //train_set.insert(p2);
    //train_set.insert(p3);
    //train_set.insert(p4);    
    //train_set.insert(p5);
    //train_set.insert(p6);
    //train_set.insert(p7);
    //train_set.insert(p8);
    //train_set.insert(p9);
    //train_set.insert(p10);

    categories_size = train_set.size();
    cout << "发现 " << categories_size << "种类别物体..." << endl;
}

Mat Mycluster(const Mat& _descriptors) {
    Mat labels, vocabulary;
    int K{ 4 }, attemps{ 100 };
    //int flags = ANN::KMEANS_RANDOM_CENTERS;
    std::vector<int> best_labels;
    double compactness_measure{ 0. };
    const int myK{ 4 }, myattemps{ 100 }, max_iter_count{ 100 };
    const double epsilon{ 0.001 };
    //ANN::kmeans<float>(_descriptors, myK, best_labels, vocabulary, compactness_measure, max_iter_count, epsilon, myattemps, flags);
    return vocabulary;
}
// 训练图片feature聚类，得出词典
void categorizer::bulid_vacab()
{
    // 以写模式打开文件
    ifstream fin("vocabulary.txt");
    //如果之前已经生成好，就不需要重新聚类生成词典
    if (fin)
    {
        cout << "图片已经聚类，词典已经存在.." << endl;
    }
    else
    {
        //存放kmeans的输入矩阵，64*提取到的特征点
        //vector<IPoint>my_vocab_descriptors;
        // 对于每一幅模板，提取SURF算子，存入到my_vocab_descriptors中
        multimap<string, Mat> ::iterator i = train_set.begin();
        for (; i != train_set.end(); i++)
        {
            Mat templ = (*i).second;
            templ.convertTo(templ, CV_32F);
            vector<IPoint> ips1 = surf.GetAllFeatures(templ);
            //将每一张图的特征点放在总的里面
            my_vocab_descriptors.insert(my_vocab_descriptors.end(), ips1.begin(), ips1.end());

        }
        cout << "训练图片开始聚类..." << endl;
        // 对ORB描述子进行聚类

        vector<vector<float>> my_data(my_vocab_descriptors.size());
        //将vector<IPinot>类型和vector<vector<float>>进行转换
        for (size_t i = 0; i < my_vocab_descriptors.size(); i++)
        {
            float *my_descriptor =my_vocab_descriptors[i].descriptor;
            vector<float> my_temp(my_descriptor, my_descriptor+64);
            my_data[i] = my_temp;
           // my_data.insert(my_data.end(), my_temp.begin(), my_temp.end());
        }
        //将所有的特征点放在属性中
        main_data = my_data;
        //使用mykmeans进行聚类
        vector<int> best_labels;
        vector<vector<float>> centers;
        double compactness_measure{ 0. };
        const int attemps{ 100 }, max_iter_count{ 100 };
        const double epsilon{ 0.001 };
        const int flags = MYKMEANS::KMEANS_RANDOM_CENTERS;
        cout << my_data[0].size() << endl;//64
    MYKMEANS::kmeans<float>(my_data, clusters, best_labels, centers, compactness_measure, max_iter_count, epsilon, attemps, flags);
        main_centers = centers;
        //vocab = bowtrainer->cluster(vocab_descriptors);
        cout << "聚类完毕，得出词典..." << endl;
        //以文件格式保存词典
        ofstream outfile;
        outfile.open("vocabulary.txt");
        for (int i = 0; i < main_centers.size(); i++) {
            for (int j = 0; j < main_centers[0].size(); j++) {
                outfile << main_centers[i][j];
                outfile << " ";
            }
            outfile << endl;
        }
        // 关闭打开的文件
        outfile.close();
    }


}


float categorizer::mydistance(vector<float>& fv, vector<float>& col)
{
    float distance = 0;
        for (size_t i = 0; i < fv.size(); ++i)
            distance += (col[i] - fv[i]) * (col[i] - fv[i]); 
        distance = sqrt(distance);
    
    return distance;
}

int categorizer::mymatch(vector<vector<float>>& input)
{
    //float dist;
    //double minDist = DBL_MAX;
    //int minIndex = 0;
    //for (size_t i = 0; i < input.size(); ++i)
    //{
    //    dist = mydistance(main_data[i],col);// main_data[i].distance(f);
    //    if (dist < minDist)
    //    {
    //        minDist = dist;
    //        minIndex = i;
    //    }
    //}
    return 0;
}
bool categorizer::getBoF(vector<vector<float>>& input,FeatureHistogram& hist, bool normalized)
{
    bool built = true;

        int idx;
        hist.resize(clusters);
        hist.zero();
        for (size_t i = 0; i < input.size(); ++i)
        {
            float dist;
            double minDist = DBL_MAX;
            int minIndex = 0;
            for (size_t j = 0; j < clusters; ++j)
            {
                dist = mydistance(input[i], main_centers[j]);// main_data[i].distance(f);
                if (dist < minDist)
                {
                    minDist = dist;
                    minIndex = j;
                }
            }
            idx = minIndex;
            hist.addAt(idx);
        }
        if (normalized)
            hist.normalize();
        return true;
    

}

//构造bag of words一幅图像就可以使用一个K维的向量表示
void categorizer::compute_bow_image()
{
    cout << "构造bag of words..." << endl;
    //如果词典存在则直接读取
    ifstream fin("vocabulary.txt");
    if (!fin) {
        cout << "vocabulary.txt文件不存在 "  << endl;
    }
    else {
        string line;
        while (getline(fin, line))
        {
            stringstream ss(line);
            float token;
            vector<float>temp;
            while (ss >> token)
            {
                temp.push_back(token);
            }
            main_centers.push_back(temp);
        }
    }
    cout << "文件读取完毕" << endl;
    ifstream hfin("model.txt");
    if (!hfin) {
        //构建直方图特征
        FeatureHistogram hist;
        SVMClassifier svm;
        multimap<string, Mat> ::iterator i = train_set.begin();
        int j = 0;
        for (; i != train_set.end(); i++)
        {
            Mat templ = (*i).second;
            templ.convertTo(templ, CV_32F);
            vector<IPoint> bof_feature = surf.GetAllFeatures(templ);
            vector<vector<float>> bof_descriptor(bof_feature.size());
            //将vector<IPinot>类型和vector<vector<float>>进行转换
            for (int i = 0; i < bof_feature.size(); i++)
            {
                float* my_bof_descriptor = bof_feature[i].descriptor;
                vector<float> my_temp(my_bof_descriptor, my_bof_descriptor + 64);
                bof_descriptor[i] = my_temp;
                // my_data.insert(my_data.end(), my_temp.begin(), my_temp.end());
            }
            getBoF(bof_descriptor, hist, true);
            int n = atoi((*i).first.c_str());
            cout << "* i first==" << endl;
            cout << (*i).first.c_str() << endl;
            cout << n << endl;
            hist.setLabel(n);
            //main_labels.push_back(n); j++;
            svm.add(hist);
        }
        cout << "train   bag of words..." << endl;
        svm_problem prob;
        prob.l = svm.size;        // 训练样本数
        prob.y = new double[categories_size];
        prob.x = new svm_node * [svm.size];
        main_labels = svm.svm_labels;
        svm_node* node = new svm_node[svm.size * (1 + svm.length)];
        for (int k = 0; k < main_labels.size(); k++) {
            prob.y[k] = main_labels[k];
            cout << k << "k=" << endl;
        }
        // 按照格式打包
        for (int i = 0; i < svm.size; i++)
        {
            for (int j = 0; j < svm.length; j++)
            {   // 看不懂指针就得复习C语言了，类比成二维数组的操作
                node[(svm.length + 1) * i + j].index = j + 1;
                node[(svm.length + 1) * i + j].value = svm.svm_trainData[i][j];
            }
            node[(svm.length + 1) * i + svm.length].index = -1;
            prob.x[i] = &node[(svm.length + 1) * i];
        }
        svm_model* svmModel;
        svm_parameter param;
        param.svm_type = C_SVC;
        param.kernel_type = RBF;
        param.degree = 3;
        param.gamma = 0.5;
        param.coef0 = 0;
        param.nu = 0.5;
        param.cache_size = 40;
        param.C = 2000;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;
        // param.probability = 0;
        param.nr_weight = 0;
        param.weight = NULL;
        param.weight_label = NULL;

        cout << "  ================================== svm_train " << endl;
        svmModel = svm_train(&prob, &param);
        svm_save_model("model.txt", svmModel);
        cout << "  ================================== svm_save_model " << endl;
    }
  
}

//训练分类器

void categorizer::trainSvm()
{


    //}
}


//对测试图片进行分类

void categorizer::category_By_svm(Mat input_pic)
{
    cout << "物体分类开始..." << endl;
    FeatureHistogram predict_hist;
    double result;
    svm_model* svmModel = svm_load_model("model.txt");
    Mat templ = BaseMat;//cv::imread("C:\\Java\\MyBOW\\data\\test_image\\100.png");
    templ.convertTo(templ, CV_32F);
    vector<IPoint> bof_feature = surf.GetAllFeatures(templ);
    vector<vector<float>> bof_descriptor(bof_feature.size());
    //将vector<IPinot>类型和vector<vector<float>>进行转换
    for (int i = 0; i < bof_feature.size(); i++)
    {
        float* my_bof_descriptor = bof_feature[i].descriptor;
        vector<float> my_temp(my_bof_descriptor, my_bof_descriptor + 64);
        bof_descriptor[i] = my_temp;
        // my_data.insert(my_data.end(), my_temp.begin(), my_temp.end());
    }
    getBoF(bof_descriptor, predict_hist, true);
    svm_node* input = new svm_node[2];
    cout << "                                                 ";
    cout << predict_hist.size << endl;//2
    for (int l = 0; l < predict_hist.data.size(); l++) {
        input[l].index = l + 1;
        input[l].value = predict_hist.data[l];
    }
    input[predict_hist.data.size()].index = -1;
    cout << "  ================================== svm_predict " << endl;
    int predictValue = svm_predict(svmModel, input);
    cout << "这幅图的类别id是： ";
    cout << predictValue << endl;// 2
    BestID = predictValue;
}



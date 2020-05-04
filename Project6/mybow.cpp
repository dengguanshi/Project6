#include "func.h"


using namespace std;
#define TRAIN_FOLDER "C:/Java/MyBOW/data/train_images/"
#define DATA_FOLDER "C:/Java/MyBOW/data/"



// �Ƴ���չ����������ģ����֯����Ŀ
string categorizer::remove_extention(string full_name)
{
    //find_last_of�ҳ��ַ����һ�γ��ֵĵط�
    int last_index = full_name.find_last_of(".");
    string name = full_name.substr(0, last_index);
    return name;
}

// ���캯��
categorizer::categorizer(int _clusters,Mat baseMat, multimap<string, Mat> train_label_mat)
{
    cout << "ִ��categorizer���캯��..." << endl;
    clusters = _clusters;
    //��ʼ��ָ��
    int minHessian = 400;
    BaseMat = baseMat;
    //��ʹ��mongdb�������ʱ�������д���
    ifstream fin("vocabulary.txt");
    //���֮ǰ�Ѿ����ɺã��Ͳ���Ҫ���¾������ɴʵ�
    if (fin)
    {
        cout << "������ֵtrain_set" << endl;
    }
    else {
        train_set = train_label_mat;//!!!
    }

    //��ȡѵ����
    make_train_set();
}

//����ѵ������
void categorizer::make_train_set()
{
    cout << "��ȡѵ����..." << endl;

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
    cout << "���� " << categories_size << "���������..." << endl;
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
// ѵ��ͼƬfeature���࣬�ó��ʵ�
void categorizer::bulid_vacab()
{
    // ��дģʽ���ļ�
    ifstream fin("vocabulary.txt");
    //���֮ǰ�Ѿ����ɺã��Ͳ���Ҫ���¾������ɴʵ�
    if (fin)
    {
        cout << "ͼƬ�Ѿ����࣬�ʵ��Ѿ�����.." << endl;
    }
    else
    {
        //���kmeans���������64*��ȡ����������
        //vector<IPoint>my_vocab_descriptors;
        // ����ÿһ��ģ�壬��ȡSURF���ӣ����뵽my_vocab_descriptors��
        multimap<string, Mat> ::iterator i = train_set.begin();
        for (; i != train_set.end(); i++)
        {
            Mat templ = (*i).second;
            templ.convertTo(templ, CV_32F);
            vector<IPoint> ips1 = surf.GetAllFeatures(templ);
            //��ÿһ��ͼ������������ܵ�����
            my_vocab_descriptors.insert(my_vocab_descriptors.end(), ips1.begin(), ips1.end());

        }
        cout << "ѵ��ͼƬ��ʼ����..." << endl;
        // ��ORB�����ӽ��о���

        vector<vector<float>> my_data(my_vocab_descriptors.size());
        //��vector<IPinot>���ͺ�vector<vector<float>>����ת��
        for (size_t i = 0; i < my_vocab_descriptors.size(); i++)
        {
            float *my_descriptor =my_vocab_descriptors[i].descriptor;
            vector<float> my_temp(my_descriptor, my_descriptor+64);
            my_data[i] = my_temp;
           // my_data.insert(my_data.end(), my_temp.begin(), my_temp.end());
        }
        //�����е����������������
        main_data = my_data;
        //ʹ��mykmeans���о���
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
        cout << "������ϣ��ó��ʵ�..." << endl;
        //���ļ���ʽ����ʵ�
        ofstream outfile;
        outfile.open("vocabulary.txt");
        for (int i = 0; i < main_centers.size(); i++) {
            for (int j = 0; j < main_centers[0].size(); j++) {
                outfile << main_centers[i][j];
                outfile << " ";
            }
            outfile << endl;
        }
        // �رմ򿪵��ļ�
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

//����bag of wordsһ��ͼ��Ϳ���ʹ��һ��Kά��������ʾ
void categorizer::compute_bow_image()
{
    cout << "����bag of words..." << endl;
    //����ʵ������ֱ�Ӷ�ȡ
    ifstream fin("vocabulary.txt");
    if (!fin) {
        cout << "vocabulary.txt�ļ������� "  << endl;
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
    cout << "�ļ���ȡ���" << endl;
    ifstream hfin("model.txt");
    if (!hfin) {
        //����ֱ��ͼ����
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
            //��vector<IPinot>���ͺ�vector<vector<float>>����ת��
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
        prob.l = svm.size;        // ѵ��������
        prob.y = new double[categories_size];
        prob.x = new svm_node * [svm.size];
        main_labels = svm.svm_labels;
        svm_node* node = new svm_node[svm.size * (1 + svm.length)];
        for (int k = 0; k < main_labels.size(); k++) {
            prob.y[k] = main_labels[k];
            cout << k << "k=" << endl;
        }
        // ���ո�ʽ���
        for (int i = 0; i < svm.size; i++)
        {
            for (int j = 0; j < svm.length; j++)
            {   // ������ָ��͵ø�ϰC�����ˣ���ȳɶ�ά����Ĳ���
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

//ѵ��������

void categorizer::trainSvm()
{


    //}
}


//�Բ���ͼƬ���з���

void categorizer::category_By_svm(Mat input_pic)
{
    cout << "������࿪ʼ..." << endl;
    FeatureHistogram predict_hist;
    double result;
    svm_model* svmModel = svm_load_model("model.txt");
    Mat templ = BaseMat;//cv::imread("C:\\Java\\MyBOW\\data\\test_image\\100.png");
    templ.convertTo(templ, CV_32F);
    vector<IPoint> bof_feature = surf.GetAllFeatures(templ);
    vector<vector<float>> bof_descriptor(bof_feature.size());
    //��vector<IPinot>���ͺ�vector<vector<float>>����ת��
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
    cout << "���ͼ�����id�ǣ� ";
    cout << predictValue << endl;// 2
    BestID = predictValue;
}



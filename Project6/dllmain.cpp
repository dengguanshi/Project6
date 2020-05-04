// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "mao_arithmatic_Arithmatic.h"
#include <opencv2/opencv.hpp>
#include"main.h"
#include"func.h"
#include <opencv2\imgproc\types_c.h>
using namespace std;
using namespace cv;
class Pic
{
public:
	String id;
	Mat picMat;
};
Pic MyJobject2Mat2(JNIEnv* env, jobject jobj1);
JNIEXPORT jstring JNICALL Java_mao_arithmatic_Arithmatic_MyBOWMatch
(JNIEnv*env, jobject , jstring test_id, jstring test_data, jobject train_list){
	multimap<string, Mat> train_label_mat;
	ifstream fin("vocabulary.txt");
	//如果之前已经生成好，就不需要重新聚类生成词典
	if (fin)
	{
		cout << "图片已经聚类，词典已经存在.." << endl;
	}
	else {
		//获取list 
		jclass cls_arraylist = env->GetObjectClass(train_list);
		//获取属性
		jmethodID arraylist_get = env->GetMethodID(cls_arraylist, "get", "(I)Ljava/lang/Object;");
		jmethodID arraylist_size = env->GetMethodID(cls_arraylist, "size", "()I");
		//获取长度
		jint len = env->CallIntMethod(train_list, arraylist_size);
		Pic PicArray;

		cout << "len" << len << endl;
		for (int i = 0; i < len; i++) {
			jobject obj_user = env->CallObjectMethod(train_list, arraylist_get, i);
			PicArray = MyJobject2Mat2(env, obj_user);
			pair<string, Mat> temp_pair(PicArray.id, PicArray.picMat);
			train_label_mat.insert(temp_pair);
			cout << "=PicArray[i]==" << PicArray.id << endl;
		}
	}

	//接收图片的字符串信息
	char const* str;
	str = env->GetStringUTFChars(test_data, 0);
	if (str == NULL) {
		return NULL;
	}
	//将字符串转换成mat
	string str_picdata = str;
	Mat mymat_input = Base2Mat(str_picdata);

	int ab = mymain(mymat_input, train_label_mat);
	// 返回一个字符串
	char const* tmpstr ="4";
	jstring rtstr = env->NewStringUTF(tmpstr);
	Mat my_mat(2,1,1);
	cout << my_mat.size() << endl;
	return rtstr;
}
Pic MyJobject2Mat2(JNIEnv* env, jobject jobj1) {
	//获得pic类引用
	jclass pic_cla = env->GetObjectClass(jobj1);
	if (pic_cla == NULL)
	{
		cout << "GetObjectClass failed \n";
	}

	//获取picture类
	jclass mypic_cls = env->GetObjectClass(jobj1);
	//通过get方法获取属性
	jmethodID label_methodId = env->GetMethodID(mypic_cls, "getLabel", "()Ljava/lang/String;");
	jmethodID picData_methodId = env->GetMethodID(mypic_cls, "getPicData","()Ljava/lang/String;");
	jstring label_jstring = (jstring)env->CallObjectMethod(jobj1, label_methodId);
	jstring picData_jstring = (jstring)env->CallObjectMethod(jobj1, picData_methodId);
	const char* label_char = env->GetStringUTFChars(label_jstring, 0);
	const char* picData_char = env->GetStringUTFChars(picData_jstring, 0);
	string label_string = label_char;
	string picData_string = picData_char;

	////获取类中的数据
	//jfieldID idFieldID = env->GetFieldID(pic_cla, "label", "Ljava/lang/String;"); //获得得Student类的属性id 
	//jfieldID picdataFieldID = env->GetFieldID(pic_cla, "picdata", "Ljava/lang/String;"); // 获得属性ID
	//if (idFieldID == NULL) {
	//	cout << "jfieldID\n";
	//}
	//
	////将类中的数据转换成string
	////1.jstring接收属性值
	//jstring id = (jstring)env->GetObjectField(jobj1, idFieldID);  //获得属性值
	//jstring picdata = (jstring)env->GetObjectField(jobj1, picdataFieldID);//获得属性值
	////2.const char*接收转换的jstring
	//const char* c_picdata = env->GetStringUTFChars(picdata, NULL);//转换成 char *
	//const char* c_id = env->GetStringUTFChars(id, NULL);//转换成 char *
	////3.const char*转换成string类型
	//string str_picdata = c_picdata;
	//string str_id = c_id;
	//cout << "cout << str_id << endl;" << endl;
	//cout << str_id << endl;
	//4.释放引用
	env->ReleaseStringUTFChars(label_jstring, label_char);
	env->ReleaseStringUTFChars(picData_jstring, picData_char);

	cout << "=======C++====" << endl;
	///*cout << " at Native age is :" << id << " # name is " << str_name << endl;*/
	///*const char* str = "C++String";*/
	//jstring rtstr = env->NewStringUTF(str);//env->NewStringUTF(c_name);

	//将传送来的字符串转换成mat图像方便进行图像处理
	Mat mymat = Base2Mat(picData_string);
	Mat testpic;
	//将使用base64转换的图像转换成灰度图处理
	cvtColor(mymat, testpic, CV_RGB2GRAY);
	Pic outputPic;
	outputPic.picMat = testpic;
	outputPic.id = label_string;
	return outputPic;
}

  

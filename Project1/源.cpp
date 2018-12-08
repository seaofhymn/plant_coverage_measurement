#include<opencv.hpp>  
using namespace std;
using namespace cv;


void grayImageShow(cv::Mat &input, cv::Mat &output)
{
	for (int i = 0; i < input.rows; ++i)
	{
		for (int j = 0; j < input.cols; ++j)
		{
			output.at<uchar>(i, j) = cv::saturate_cast<uchar>(-0.311 * input.at<cv::Vec3b>(i, j)[0] + 1.262 * input.at<cv::Vec3b>(i, j)[1] - 0.884 * input.at<cv::Vec3b>(i, j)[2]);
			//output.at<uchar>(i, j) = cv::saturate_cast<uchar>(-1 * input.at<cv::Vec3b>(i, j)[0] + 2 * input.at<cv::Vec3b>(i, j)[1] - 1 * input.at<cv::Vec3b>(i, j)[2]);
		}
	}
	cv::imshow("dst", output);
	imwrite("gray.bmp", output);
}

int bSums(Mat src)
{

	int counter = 0;
	//�������������ص�  
	Mat_<uchar>::iterator it = src.begin<uchar>();
	Mat_<uchar>::iterator itend = src.end<uchar>();
	for (; it != itend; ++it)
	{
		if ((*it)>0) counter += 1;//��ֵ�������ص���0����255  
	}
	return counter;
}

int Otsu(IplImage* src)
{
	int height = src->height;
	int width = src->width;
	long size = height * width;

	//histogram    
	float histogram[256] = { 0 };
	for (int m = 0; m < height; m++)
	{
		unsigned char* p = (unsigned char*)src->imageData + src->widthStep * m;
		for (int n = 0; n < width; n++)
		{
			histogram[int(*p++)]++;
		}
	}

	int threshold;
	long sum0 = 0, sum1 = 0; //�洢ǰ���ĻҶ��ܺͺͱ����Ҷ��ܺ�  
	long cnt0 = 0, cnt1 = 0; //ǰ�����ܸ����ͱ������ܸ���  
	double w0 = 0, w1 = 0; //ǰ���ͱ�����ռ����ͼ��ı���  
	double u0 = 0, u1 = 0;  //ǰ���ͱ�����ƽ���Ҷ�  
	double variance = 0; //�����䷽��  
	int i, j;
	double u = 0;
	double maxVariance = 0;
	for (i = 1; i < 256; i++) //һ�α���ÿ������  
	{
		sum0 = 0;
		sum1 = 0;
		cnt0 = 0;
		cnt1 = 0;
		w0 = 0;
		w1 = 0;
		for (j = 0; j < i; j++)
		{
			cnt0 += histogram[j];
			sum0 += j * histogram[j];
		}

		u0 = (double)sum0 / cnt0;
		w0 = (double)cnt0 / size;

		for (j = i; j <= 255; j++)
		{
			cnt1 += histogram[j];
			sum1 += j * histogram[j];
		}

		u1 = (double)sum1 / cnt1;
		w1 = 1 - w0; // (double)cnt1 / size;  

		u = u0 * w0 + u1 * w1; //ͼ���ƽ���Ҷ�  
		printf("u = %f\n", u);
		//variance =  w0 * pow((u0 - u), 2) + w1 * pow((u1 - u), 2);  
		variance = w0 * w1 *  (u0 - u1) * (u0 - u1);
		if (variance > maxVariance)
		{
			maxVariance = variance;
			threshold = i;
		}
	}

	printf("threshold = %d\n", threshold);
	return threshold;
}






int main(void)
{
	int Otsu(IplImage* src);
	cv::Mat src, gray, dst,aa;
	gray = cv::imread("΢��ͼƬ_20180606210218.bmp", cv::IMREAD_GRAYSCALE);//��imread()�õ��ĻҶ�ͼ��  
	src = cv::imread("΢��ͼƬ_20180606210218.bmp");
	dst.create(src.rows, src.cols, CV_8UC1);

	cv::imshow("scr", src);
	cv::imshow("gray", gray);
	grayImageShow(src, dst);
	//aa=cv::imread("gray.bmp");
	//cvNamedWindow("�����㷨", 0);
	//cvShowImage("�����㷨",&aa);


	//Mat a(dst);
	IplImage* img = cvLoadImage("gray.bmp", 0);
	IplImage* dst2 = cvCreateImage(cvGetSize(img), 8, 1);
	int threshold = Otsu(img);
	cvThreshold(img, dst2, threshold, 255, CV_THRESH_BINARY);
	cvNamedWindow("ostu",0);
	cvShowImage("ostu", dst2);
	cvSaveImage("ostu.bmp", dst2);
	



	Mat image = imread("ostu.bmp");

	//Mat Salt_Image;
	//image.copyTo(Salt_Image);


	Mat image2, image3, image4,image5,image6,image7,image8;
	
	medianBlur(image, image2, 9);
	medianBlur(image2, image3, 9);
	medianBlur(image3, image4, 9);
	medianBlur(image4, image5, 9);
	medianBlur(image5, image6, 9);
	medianBlur(image6, image7, 9);
	medianBlur(image7, image8, 9);
	//aa = cv::imread("ostu.bmp");
	//cvNamedWindow("�˲�", 0);
	//cvShowImage("�˲�", &aa);
	
	imshow("��ֵ�˲�", image2);
	imshow("��ֵ�˲�2", image3);
	imshow("��ֵ�˲�3", image4);

	imwrite("��ֵ�˲�.bmp", image2);
	imwrite("��ֵ�˲�2.bmp", image3);
	imwrite("��ֵ�˲�3.bmp", image4);
	imwrite("��ֵ�˲�4.bmp", image5);
	imwrite("��ֵ�˲�5.bmp", image6);
	imwrite("��ֵ�˲�6.bmp", image7);
	imwrite("��ֵ�˲�7.bmp", image8);




	//Mat a1 = imread("��ֵ�˲�5.bmp");

	//int a = bSums(a1);//���ú���bSums  
	//imshow("A", a1);
	//cout << "A:" << a;






	waitKey(-1);
}







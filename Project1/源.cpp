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
	//迭代器访问像素点  
	Mat_<uchar>::iterator it = src.begin<uchar>();
	Mat_<uchar>::iterator itend = src.end<uchar>();
	for (; it != itend; ++it)
	{
		if ((*it)>0) counter += 1;//二值化后，像素点是0或者255  
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
	long sum0 = 0, sum1 = 0; //存储前景的灰度总和和背景灰度总和  
	long cnt0 = 0, cnt1 = 0; //前景的总个数和背景的总个数  
	double w0 = 0, w1 = 0; //前景和背景所占整幅图像的比例  
	double u0 = 0, u1 = 0;  //前景和背景的平均灰度  
	double variance = 0; //最大类间方差  
	int i, j;
	double u = 0;
	double maxVariance = 0;
	for (i = 1; i < 256; i++) //一次遍历每个像素  
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

		u = u0 * w0 + u1 * w1; //图像的平均灰度  
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
	gray = cv::imread("微信图片_20180606210218.bmp", cv::IMREAD_GRAYSCALE);//由imread()得到的灰度图像  
	src = cv::imread("微信图片_20180606210218.bmp");
	dst.create(src.rows, src.cols, CV_8UC1);

	cv::imshow("scr", src);
	cv::imshow("gray", gray);
	grayImageShow(src, dst);
	//aa=cv::imread("gray.bmp");
	//cvNamedWindow("超绿算法", 0);
	//cvShowImage("超绿算法",&aa);


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
	//cvNamedWindow("滤波", 0);
	//cvShowImage("滤波", &aa);
	
	imshow("中值滤波", image2);
	imshow("中值滤波2", image3);
	imshow("中值滤波3", image4);

	imwrite("中值滤波.bmp", image2);
	imwrite("中值滤波2.bmp", image3);
	imwrite("中值滤波3.bmp", image4);
	imwrite("中值滤波4.bmp", image5);
	imwrite("中值滤波5.bmp", image6);
	imwrite("中值滤波6.bmp", image7);
	imwrite("中值滤波7.bmp", image8);




	//Mat a1 = imread("中值滤波5.bmp");

	//int a = bSums(a1);//调用函数bSums  
	//imshow("A", a1);
	//cout << "A:" << a;






	waitKey(-1);
}







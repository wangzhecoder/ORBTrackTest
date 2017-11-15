#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

int main(int argc, char ** argv)
{
    cv::Mat img_1,img_2;
    img_1 = cv::imread(argv[1]);
    img_2 = cv::imread(argv[2]);
    if (!img_1.data || !img_2.data)  
    {  
        cout << "error reading images " << endl;  
        return -1;  
    }
    cv::Ptr<cv::FeatureDetector> orb =
				cv::FeatureDetector::create("ORB");
    std::vector<cv::KeyPoint> keyPoints_1, keyPoints_2;
    cv::Mat descriptors_1, descriptors_2; 
    orb->detect(img_1,keyPoints_1,cv::Mat());  
    orb->detect(img_2,keyPoints_2,cv::Mat()); 
    cv::Ptr<cv::DescriptorExtractor> descriptor_extractor =
                    cv::DescriptorExtractor::create("ORB"); 
    descriptor_extractor->compute(img_1,keyPoints_1,descriptors_1);
    descriptor_extractor->compute(img_2,keyPoints_2,descriptors_2);

    cv::Ptr<DescriptorMatcher> matcher = 
               cv::DescriptorMatcher::create("BruteForce-Hamming");

    std::vector<DMatch> matches;  
    matcher->match(descriptors_1, descriptors_2, matches);

    double max_dist = 0; double min_dist = 100; 

    for( int i = 0; i < descriptors_1.rows; i++ )  
    {   
        double dist = matches[i].distance;  
        if( dist < min_dist ) min_dist = dist;  
        if( dist > max_dist ) max_dist = dist;  
    }  
    cout<<"-- Max dist : "<<max_dist<<endl;  
    cout<<"-- Min dist : "<<min_dist<<endl;
    //-- Draw only "good" matches (i.e. whose distance is less than 0.6*max_dist )  
    //-- PS.- radiusMatch can also be used here.  
    std::vector< DMatch > good_matches;  
    for( int i = 0; i < descriptors_1.rows; i++ )  
    {   
        if( matches[i].distance < 0.6*max_dist )  
        {   
            good_matches.push_back( matches[i]);   
        }  
    }  
    Mat img_matches;  
    drawMatches(img_1, keyPoints_1, img_2, keyPoints_2,  
        good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),  
        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  
    imshow( "Match", img_matches);  
    cvWaitKey();  
    return 0;
}
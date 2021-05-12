#ifndef WORKERS_CPU_OPTRONICSWORKER_H
#define WORKERS_CPU_OPTRONICSWORKER_H

/*
#define ImgResolutionX 8712 //Image Resolution  
#define ImgResolutionY 5813 //Image Resolution 
#define NumofFeaturesColour 10 		//Number of bins in the feature histogram
#define NumofFeaturesGradient 10	//Number of bins in the colour histogram
#define TargetImgWidth 10 	//Assume a candidate target appears in a box with 10 x 10 pixels in an image
*/
#define Set_Yk_sizeMax 63	//Assume the maximum number of target candidates is 63
#define Ydimension 5   //The size of the image processing output for one candidate target containing [x-coordin. y-coordin., R, G, B]
#define M_PI 3.14159265358979323846  // the value of pi

#include <core/component.h>
#include <core/worker.h>

class OptronicsWorker: public Worker{
    public:
	    OptronicsWorker(const BehaviourContainer& container, const std::string& inst_name);
        ~OptronicsWorker();
        void RunTest (); 
		const std::string &get_version() const { return version; }
		
		
		void NewImage(int IRX, int IRY, int NFC, int NFG, int TIW, int NC);
		void ClearImage(void);
		void init(double dist, double angle);
		void GenerateMeasurements(void);
		double SetofMeasurement(int Seti, int yi);
		int NumofMeasurements(void);
		
		
    private:
	    const std::string version = "1.0.1";

		int ImgResolutionX; //Image Resolution  
		int ImgResolutionY; //Image Resolution 
		int NumofFeaturesColour; 		//Number of bins in the feature histogram
		int NumofFeaturesGradient;	//Number of bins in the colour histogram
		int TargetImgWidth; 	//Assume a candidate target appears in a box with 10 x 10 pixels in an image

		int Set_Yk_size;
		double Set_Yk[Set_Yk_sizeMax][Ydimension];

        double *x;		
		int *input_img;
		int NumofTargets;
		
		int NumofCameras;
		int *CamBin;


		
		void HSITransform (int *original_img, float *output_img);
		void Orientation (float *input_img, float *gradient_img);
		void Histogram (int m, double *FeatureMap, int nh, int h, double *x, double *y, float *img, double *P);				

//		void ImageProcessing(int *input_img, int NumofTargets, int *Set_Yk_size, double *Set_Yk);
		int ImageProcessing(int CameraNum);
		
		
		
};

#endif
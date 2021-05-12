#include "optronicsworker.h"
#include <core/component.h>
#include <pthread.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <assert.h>
#include <thread>         
#include <mutex>  
#include <random>  
#include <stdlib.h> 


OptronicsWorker::OptronicsWorker(const BehaviourContainer& container, const std::string& inst_name) : Worker(container, GET_FUNC, inst_name){


}

OptronicsWorker::~OptronicsWorker(){
}


void OptronicsWorker::RunTest() {
}

void OptronicsWorker::NewImage(int IRX, int IRY, int NFC, int NFG, int TIW, int NC){

    unsigned long long ll;

	ImgResolutionX = IRX; 			//Image Resolution  
	ImgResolutionY = IRY; 			//Image Resolution 
	NumofFeaturesColour =  NFC; 	//Number of bins in the feature histogram
	NumofFeaturesGradient = NFG;	//Number of bins in the colour histogram
	TargetImgWidth = TIW;			//Number of pixels of a target box's edge
	NumofCameras = NC; 				//Number of Cameras
	
    NumofTargets = 0;

// Allocate memory to store an image.
	ll = (unsigned long long) ImgResolutionX * ImgResolutionY * 3;  //the amount of memory required for an image
    input_img = (int *)malloc(ll * sizeof(int)); 

	ll = (unsigned long long) TargetImgWidth*TargetImgWidth*2 * sizeof(double);
	x = (double *) malloc(ll);  
	
	CamBin	= (int *) malloc(NumofCameras * sizeof(int));  

	if (input_img == NULL) // fail to allocate memory
		printf("input_img : Null \n");	

	for (int i = 0; i < NumofCameras; i++) { 
		CamBin[i] = 0;
	}

    for (ll = 0; ll < TargetImgWidth*TargetImgWidth*2; ll++) {
         x[ll] = 1;
    }

	for (int i = 0; i < ImgResolutionX; i++) { 
		for (int j = 0; j < ImgResolutionY; j++) { 
			for (int k = 0; k < 3; k++) { 
				input_img[i*ImgResolutionY*3 + j* 3 + k] = 100; // Initialise the image with RGB = 100.
			}
		}
	}
	
}

void OptronicsWorker::ClearImage() {
// Free the memory for storing an image in memory
// It should be called after image processing is done.

	free(input_img);
	free(x);	
	free(CamBin);
}

void OptronicsWorker::init(double dist, double angle) {
// Insert a dot in an image at the positiong according to the inputs [dist, angle]. 

	
	int i, j, k;
	

	//X coordinate of an object in an image. Unit is pixel. 
	i = nearbyint(dist * cos(angle / 180 * M_PI) / 10 * (ImgResolutionX/2) + (ImgResolutionX/2));
	
	//Y coordinate of an object in an image. Unit is pixel.
	j = nearbyint(dist * sin(angle / 180 * M_PI) / 10 * (ImgResolutionY/2) + (ImgResolutionY/2));
	
	if (i < 0)
		i = 0;
	if (i >= ImgResolutionX)
		i = ImgResolutionX-1;
	if (j < 0)
		j = 0;
	if (j >= ImgResolutionY)
		j = ImgResolutionY-1;

	// The RGB values of a target candidate
	input_img[i*ImgResolutionY*3 + j* 3 + 0] = 201;
	input_img[i*ImgResolutionY*3 + j* 3 + 1] = 202;
	input_img[i*ImgResolutionY*3 + j* 3 + 2] = 203;

	// The measurement of a candiate target. [x-coordin. y-coordin., R, G, B]
	Set_Yk[NumofTargets][0] = i; 
	Set_Yk[NumofTargets][1] = j; 
	Set_Yk[NumofTargets][2] = input_img[i*ImgResolutionY*3 + j*3 + 0]; 
	Set_Yk[NumofTargets][3] = input_img[i*ImgResolutionY*3 + j*3 + 1]; 
	Set_Yk[NumofTargets][4] = input_img[i*ImgResolutionY*3 + j*3 + 2]; 

	//A new object has been registered
	NumofTargets = NumofTargets+1; 
	
	//A new object is caught by Camera k
	k = (int) floor((angle+180) / (360/NumofCameras) );
	++CamBin[k];
	
	
	printf("ImgProc - added obj. #%d at |%d|%d| in Camera %d\n", NumofTargets, i, j, k);	
}


void OptronicsWorker::GenerateMeasurements(void) {
//	Call the image processing functions and check if there is any error
  
	int i, j;
	
printf("ImgProc - GenerateMeasurements - begin \n");
printf("ImgProc - Camerabin: ");
	for (j = 0; j < NumofCameras; j++) {
printf("|%d| \t", CamBin[j]);		
	}
printf("\n");	

	for (j = 0; j < NumofCameras; j++) {
		i = ImageProcessing(j); //Call the image processing function.
	}
	if (i == 1) {
	    Set_Yk_size = 0;  // Error has been detected
	} else {
	    Set_Yk_size = NumofTargets;  // Targets are successfully detected
	}
}

int OptronicsWorker::NumofMeasurements(void) {
// Return the number of target candidate detected

    return(Set_Yk_size);
}

double OptronicsWorker::SetofMeasurement(int Seti, int yi) {
// For Measurement $Seti$, return the parameter 0 <= yi <= 4 in this format [x-coordin. y-coordin., R, G, B]
    return(Set_Yk[Seti][yi]);
}

//void OptronicsWorker::ImageProcessing(int *original_img, int NumofTargets, int *Set_Yk_size, double *Set_Yk) {
int OptronicsWorker::ImageProcessing(int CameraNum) {
// Emulate the computational load of some image processing techniques

    int i, j;
    float *output_img, *gradient_img;
    unsigned long long ll;
    double FeatureMapC[NumofFeaturesColour][3], FeatureMapG[NumofFeaturesGradient][3];
    double PColour[NumofFeaturesColour], PGradient[NumofFeaturesGradient];
    double y[2];
    

    ll = (unsigned long long) ImgResolutionX * ImgResolutionY * 3 * sizeof(float);  //the amount of memory required for an image
//printf("ll: %llu \n", ll);

    output_img = (float *) malloc(ll);
if (output_img == NULL) {
    return(1); 		//failed to allocation memory
//    printf("output_img : Null \n");
}
    
	gradient_img = (float *) malloc(ll);
if (gradient_img == NULL) {
    return(1);		//failed to allocation memory
//    printf("gradient_img : Null \n");
}
 
//printf("input_img: %p |\t output_img : %p |\t gradient_img : %p \n", input_img, output_img, gradient_img);



    HSITransform (input_img, output_img); // Colour transform for the image 
printf("After HSITransform \n");

    Orientation(output_img, gradient_img);	// Calculate the gradient and orientation
printf("After Orientation \n");

/*    for (i = 0; i < (ImgResolutionX); i++) {
    	for (j = 0; j < (ImgResolutionY); j++) {
    	    output_img[i*ImgResolutionY*3+j*3 + 0] = 100;
    	    output_img[i*ImgResolutionY*3+j*3 + 1] = 110; 
    	    output_img[i*ImgResolutionY*3+j*3 + 2] = 120; 
    	}
    }
*/


    
//   for (i = 0; i < NumofTargets; i++) { 
	for (i = 0; i < CamBin[CameraNum]; i++) { 
        /*  According to output_img and gradient_img, part of the image indicated by x and y is considered as target candidate.  
            The following functions prepare the probabilities for checking if this part of image should be considered as a target
        */
printf("i = %d \n", i);        

        Histogram (NumofFeaturesColour, &FeatureMapC[0][0], TargetImgWidth*TargetImgWidth, 1, x, y, output_img, PColour); // based on colour
printf("Histogram 1 \n");

        Histogram (NumofFeaturesGradient, &FeatureMapG[0][0], TargetImgWidth*TargetImgWidth, 1, x, y, gradient_img, PGradient); //based on gradient and orientation
printf("Histogram 2 \n");

        /* According to the probabilities PColour and PGradient, if a target candidate is considered as a target, 
           both Set_Yk_size and Set_Yk are updated.
        */
        
   }
    
	// free the memory
    free(output_img);	
    free(gradient_img);

    return(0);
}

void OptronicsWorker::HSITransform (int *original_img, float *output_img) {
//Colour space transform using HSI model
//original_img stores image in RGB values. output_img stores HSI.    
    unsigned long long i, j;
    
    for (i = 0; i < ImgResolutionX; i++) {
    	for (j = 0; j < ImgResolutionY; j++) {
			
			// Value of H
    		output_img[i*ImgResolutionY*3+j*3] = atan ((sqrt(3) * (original_img[i*ImgResolutionY*3+ j*3+ 1] - original_img[i*ImgResolutionY*3+ j*3+ 2])) / (2*original_img[i*ImgResolutionY*3+ j*3] - original_img[i*ImgResolutionY*3+ j*3+ 1] - original_img[i*ImgResolutionY*3+ j*3+ 2]) ) / M_PI * 180;
			
			// Value of I
			output_img[i*ImgResolutionY*3 + j*3 + 2] = (original_img[i*ImgResolutionY*3+ j*3] + original_img[i*ImgResolutionY*3+ j*3+ 1] + original_img[i*ImgResolutionY*3+ j*3+ 2]  )/3;
			
			// Value of S
    		if (original_img[i*ImgResolutionY*3+ j*3] < original_img[i*ImgResolutionY*3+ j*3+ 1]) {
    			if (original_img[i*ImgResolutionY*3+ j*3] < original_img[i*ImgResolutionY*3+ j*3+ 2]) {
    				output_img[i*ImgResolutionY*3+ j*3+ 1] = original_img[i*ImgResolutionY*3+ j*3];
    			} else {
    				output_img[i*ImgResolutionY*3+ j*3+ 1] = original_img[i*ImgResolutionY*3+ j*3+ 2];
    			}
    		} else {
    			if (original_img[i*ImgResolutionY*3+ j*3+ 1] < original_img[i*ImgResolutionY*3+ j*3+ 2]) {
    				output_img[i*ImgResolutionY*3+ j*3+ 1] = original_img[i*ImgResolutionY*3+ j*3+ 1];
    			} else {
    				output_img[i*ImgResolutionY*3+ j*3+ 1] = original_img[i*ImgResolutionY*3+ j*3+ 2];
    			}
    		
    		}

			if (output_img[i*ImgResolutionY*3+ j*3 + 2] != 0) {
				output_img[i*ImgResolutionY*3 + j*3 + 1] =  1 - output_img[i*ImgResolutionY*3+ j*3+ 1] / output_img[i*ImgResolutionY*3+ j*3 + 2];
			} else {
				output_img[i*ImgResolutionY*3 + j*3 + 1] =  -999999;
			}
    		
    		
    	}
    }

}



void OptronicsWorker::Orientation (float *input_img, float *gradient_img) {
//input_img is the output from HSITransform
//gradient_img[..][..][0]: the magnitude of the gradient
//gradient_img[..][..][1]: the orientation
//gradient_img[..][..][2]: reserved. This field is added such that the same function Histogram can handle the outputs from both HSITransform and Orientation.


    double dIdu, dIdv;
    int u, v, urange[2], vrange[2];
    int Su[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    int Sv[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    
    for (int i = 0; i <= (ImgResolutionX-1); i++) {
    	for (int j = 0; j <= (ImgResolutionY-1); j++) {
    		
    		//2-D Convolution for dIdu and dIdv (begin)
    		
			// calculate the range as the point may appear at the edge of an image
			urange[0] =  (i == 0);
    		urange[1] =  2 - (i == ImgResolutionX-1);
    		vrange[0] =  (j == 0);
    		vrange[1] =  2 - (j == ImgResolutionY-1);

//printf("i=%d, j=%d, urange[0]=%d, urange[1]=%d, vrange[0]=%d, vrange[1]=%d\n", i, j, urange[0], urange[1], vrange[0], vrange[1]);
    		dIdu = 0;
    		dIdv = 0;
    
    		for (u = urange[0]; u <= urange[1]; u++) {
    			for (v = vrange[0]; v <= vrange[1]; v++) {
    				dIdu += Su[v][u] * input_img[(i + u - 1)*ImgResolutionY*3 + (j + v - 1)*3 + 2];
    				dIdv += Sv[v][u] * input_img[(i + u - 1)*ImgResolutionY*3 + (j + v - 1)*3 + 2];
    			}
    
    		}
    		//2-D Convolution for dIdu and dIdv (end)
//printf("i=%d, j=%d, dIdu=%f, dIdv=%f\n", i, j, dIdu, dIdv);

    		gradient_img[i*ImgResolutionY*3 + j*3 + 0] = sqrt(dIdu * dIdu + dIdv * dIdv);
    		
    		if (dIdv == 0) {
    		    gradient_img[i*ImgResolutionY*3 + j*3 + 1] = M_PI/2;
    		} else {
    		    gradient_img[i*ImgResolutionY*3 + j*3 + 1] = atan(dIdu / dIdv);
    		}
			
			gradient_img[i*ImgResolutionY*3 + j*3 + 2] = 0; //reserved 
    	}
    }
	
}


void OptronicsWorker::Histogram (int m, double *FeatureMap, int nh, int h, double *x, double *y, float *img, double *P) {
/*
Calculate the histogram for feature extraction

m: number of features
FeatureMap: database of features which is a m x 3 matrix
nh: the number of normalised pixels of the target candidate
h: the bandwidth of kernel profile
x: normalised pixel locations of the target candidate
y: normalised pixel locations of the center of the target candidate
img: the image after transform
P: 	the probability of the features
*/
    
    int u;
    double Ch = 0;
    int b = 0;
    int i, j, v, xi1, xi2, OldScore, NewScore;
	int map[m];

	for (i = 0; i < nh; i++) {
		Ch += 1/ ( (pow(y[0] - x[i*2  ], 2) + pow(y[1] - x[i*2 + 1], 2)) / (h*h)); // the convex monotonic decreasing function k is taken as reciprocal here
	}
	Ch = 1/Ch; //i.e., $C_h$ 

	for (u = 0; u < m; u++) { 
		// the probability of the feature u is the target candidate
			P[u] = 0;
	}


	for (i = 0; i < nh; i++) {
		xi1 = x[i*2 ];     // xi stores the coordinates of the target candidate
		xi2 = x[i*2 + 1];
		
		b = 0;
		OldScore=99999999;
		NewScore=0;
		for (v = 0; v < m; v++) {
			for (j = 0; j < 3; j++) { // for RGB 

				NewScore += abs(FeatureMap[v*3 + j] - img[xi1*ImgResolutionY*3 + xi2*3 + j]);
			}
			if (NewScore < OldScore) { // find the smallest score, i.e., the smallest difference with the features in database
				OldScore = NewScore;
				b = v; // store the feature number with the smallest difference
			}
			
		}
		
		// the probability of the feature b in the target candidate
		P[b] += Ch * 1/ ( pow(y[1] - xi1, 2) + pow(y[2] - xi2, 2) / (h*h)); // the convex monotonic decreasing function k is taken as reciprocal here.
	}

}
#include "../apso_cuda.h"
#include <stdexcept>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <iostream>
#include <cstdlib>
#include <assert.h>
#include <cusolverDn.h>
#include <curand.h>
#include <math.h>

/*
    Macros to check CUDA errors.
*/

#  define CUDA_SAFE_CALL(call) {                                             \
    cudaError_t err = call;                                                  \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        throw thrust::system_error(call, thrust::cuda_category()); 			 \
    } }
	
#  define CUBLAS_SAFE_CALL(call) {                                           \
cublasStatus_t err = call;                                               \
if( CUBLAS_STATUS_SUCCESS != err) {                                      \
	fprintf(stderr, "CuBlass error in file '%s' in line %i : %d.\n",     \
			__FILE__, __LINE__, err );                                   \
	throw thrust::system_error(call, thrust::cuda_category()); 			 \
} }

#  define CUSOLVER_SAFE_CALL(call) {                                           \
    cusolverStatus_t  err = call;                                               \
    if( CUSOLVER_STATUS_SUCCESS != err) {                                      \
        fprintf(stderr, "CuSolver error in file '%s' in line %i : %d.\n",     \
                __FILE__, __LINE__, err );                                   \
        throw thrust::system_error(call, thrust::cuda_category()); 			 \
    } }
	
#define CURAND_SAFE_CALL(call) do { if((call)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

/*
   Converts 2D matrix indices to CblasColMajor storage format (linear) index
*/

#define Idx2C(i,j,ld) ((j)*(ld) + (i))

/*
   Rounds a float/double value to 2 decimal places
*/
#define nearest(x) round(x * 100) / 100

# define BLOCK_DIM_X 8
# define BLOCK_DIM_Y 8

//__global__ void initIdentityGPU(float devMatrix[BLOCK_DIM_X][BLOCK_DIM_Y], int numR, int numC) {
__global__ void initIdentityGPU(float **devMatrix, int numR, int numC) {
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if(y < numR && x < numC) {
          if(x == y)
              devMatrix[y][x] = 1.f;
          else
              devMatrix[y][x] = 0.f;
    }
}

struct expf_func{
    __device__ 
    double operator()(float x){
    return expf(x);
    }
};

struct cosf_func{
    __device__ 
    double operator()(float x){
    return cosf(x);
    }
};

namespace {  

	class RunAPSO {
	public:
		RunAPSO(int p, int N, int T, int randomNumComplexityConst) {
			p_loc = p;
			N_loc = N;
			T_loc = T;
			randomNumComplexityConst_loc = randomNumComplexityConst;
		};

		~RunAPSO() {	
           // clean up
		   // thrust vectors are automatically deleted on return	   
		};
		
		/*
		Executes the PSO particle filter algorithm using measurements.

		Input Parameters:
		  None
			
		Output Parameters:
		   None
		*/		
		void RunOnMeasurements() {
		   try {
				
				for(int t=0; t<T_loc; t++) 
				{
					thrust::host_vector<float> weightVec = CalculateAdaptiveWeights(N_loc, T_loc, t);
					thrust::host_vector<float> velocityVec = UpdateVelocity(N_loc, weightVec);
					UpdatePosition(N_loc, velocityVec);
					weightVec.clear(); 	
					velocityVec.clear();
				}	
				
				PFTimeUpdate(p_loc, N_loc, T_loc, randomNumComplexityConst_loc);
				
			}
			catch(std::exception &e)
			{
			  std::cerr << e.what();
			}
			catch (...) {
			  printf("RunOnMeasurements::unknown error");
			}
		};
		
		thrust::host_vector<float> CalculateAdaptiveWeights(int N, int T, int step){
			std::exception_ptr eptr;
			int C1 = 2;
			int C2 = 2;
			float W = 5.0f;
			float Rk = 4.5f;
			cublasHandle_t handle;
			thrust::host_vector<float> result(N,0.0f);					
			
			try {
					
 				   // cuBLAS handle creation
				   CUBLAS_SAFE_CALL(cublasCreate(&handle));
			
				   // host side vectors
				   thrust::host_vector<float> Xt(N, 2.0f);		
				   thrust::host_vector<float> Xp(N, 1.0f);		
				   thrust::host_vector<float> DXp(N*N,0.0f);
				   thrust::host_vector<float> temp1(N,0.0f);		
				   thrust::host_vector<float> temp2(N,1.0f);		
				   thrust::host_vector<float> temp3(N,1.0f);		
				   thrust::host_vector<float> temp4(N,0.0f);		
				   thrust::host_vector<float> temp5(N,0.0f);		
				   thrust::host_vector<float> temp6(N,0.0f);
				   thrust::host_vector<float> temp7(N,0.0f);				   
				   
				   // device side equivalent
				   thrust::device_vector<float> d_Xt = Xt;
				   thrust::device_vector<float> d_Xp = Xp;
				   thrust::device_vector<float> d_Dxp = DXp;
				   thrust::device_vector<float> d_temp1 = temp1;
				   thrust::device_vector<float> d_temp2 = temp2;
				   thrust::device_vector<float> d_temp3 = temp3;
				   thrust::device_vector<float> d_temp4 = temp4;
				   thrust::device_vector<float> d_temp5 = temp5;
				   thrust::device_vector<float> d_temp6 = temp6;
				   thrust::device_vector<float> d_temp7 = temp7;
				   
				   float alpha = -1.0f;
				   float beta = 0.0f;
				   
				   int T_max = T;
				   float W_ini = 0.3f;
				   float W_min = 0.1f;
				   float F_max = 10.0f;
				   
				   
				   // step 1: evaluate fitness function			   
				   thrust::host_vector<float> fitnessVal = EvaluateFitnessFunctionVector(N, Rk, Xp, Xt);
				   
				   // step 2: calculate adaptive weights using particle fitness func value
				   // 2.1 A=(1 + cos(t*Pi/t_max)) - result in temp2
				   float step_value = (step*M_PI/T_max);
				   thrust::host_vector<float> stepVec(N, step_value);
				   thrust::copy(stepVec.begin(), stepVec.end(), d_temp1.begin());
				   thrust::transform(d_temp1.begin(), d_temp1.end(), d_temp7.begin(), cosf_func());				   
				   
				   alpha = 1.0f;	
				   CUBLAS_SAFE_CALL(cublasSaxpy(handle, N, &alpha, thrust::raw_pointer_cast(d_temp7.data()), 1,
									thrust::raw_pointer_cast(d_temp2.data()), 1));
				   
				   // 2.2 B=(1- f/f_max) - result in temp3
				   alpha = (-1.f/F_max);
				   thrust::device_vector<float> d_fitnessVal = fitnessVal;
				   CUBLAS_SAFE_CALL(cublasSaxpy(handle, N, &alpha, thrust::raw_pointer_cast(d_fitnessVal.data()), 1,
									thrust::raw_pointer_cast(d_temp3.data()), 1));
				   
				   // 2.3 C=A*B*w_ini/2 + w_min
				   // setup diagonal matrix Diag(B)
				   for(int i=0; i<N; i++) {
					   DXp[Idx2C(i,i,N)] = temp3[i];
				   }	
				   // C = (w_ini/2)*(Diag(B))*A + w_min
				   alpha = (W_ini/2);
				   beta = 1.0f;
				   thrust::fill(d_temp6.begin(), d_temp6.end(), W_min);
				   CUBLAS_SAFE_CALL(cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha, 
				                    thrust::raw_pointer_cast(d_Dxp.data()), N, thrust::raw_pointer_cast(d_temp2.data()), 1, &beta, thrust::raw_pointer_cast(d_temp6.data()), 1));
				 
					// copy back result
					result = d_temp6;
			}
			catch(std::exception &e) {
				eptr = std::current_exception();
			}
			catch(...) {
				eptr = std::current_exception();
			}
			
			//cleanup
			// thrust vectors cleaned up automatically
			CUBLAS_SAFE_CALL(cublasDestroy(handle));
			
			if(eptr)
			{
				std::rethrow_exception(eptr);
			}
			
			return result;
		}
		
		thrust::host_vector<float> UpdateVelocity(int N_loc, thrust::host_vector<float> weightVec)
		{
			std::exception_ptr eptr;
			cublasHandle_t handle;
			int N = N_loc;
			thrust::host_vector<float> result(N, 0.0f);
				   
			try {	
 				   // cuBLAS handle creation
				   CUBLAS_SAFE_CALL(cublasCreate(&handle));
				   int N = N_loc;
			
				   // host side vectors
				   thrust::host_vector<float> Xt(N, 2.0f);
				   thrust::host_vector<float> Xp(N, 1.0f);
				   thrust::host_vector<float> DXp(N*N, 0.0f);
				   thrust::host_vector<float> DW(N*N, 0.0f);
				   thrust::host_vector<float> DX(N*N, 0.0f);
				   thrust::host_vector<float> DC1(N*N, 0.0f);
				   thrust::host_vector<float> DC2(N*N, 0.0f);
				   thrust::host_vector<float> DC3(N*N, 0.0f);
				   thrust::host_vector<float> Rp(N, 3.0f);
				   thrust::host_vector<float> Rt(N, 2.0f);
				   thrust::host_vector<float> Rs(N, 0.0f);				   
				   thrust::host_vector<float> Vt(N, 0.0f);
				   thrust::host_vector<float> Pt(N, 0.0f);
				   thrust::host_vector<float> Gt(N, 0.0f);				   
				   thrust::host_vector<float> R1(N, 0.0f);
				   thrust::host_vector<float> R2(N, 0.0f);
				   thrust::host_vector<float> R3(N, 0.0f);
				   
				   thrust::host_vector<float> temp1(N, 0.0f);
				   thrust::host_vector<float> temp2(N, 0.0f);
				   thrust::host_vector<float> temp3(N, 0.0f);
				   thrust::host_vector<float> temp4(N, 0.0f);
				   thrust::host_vector<float> temp5(N, 0.0f);
				   thrust::host_vector<float> temp6(N, 0.0f);
				   thrust::host_vector<float> temp7(N, 0.0f);
				   thrust::host_vector<float> temp8(N, 0.0f);
				   thrust::host_vector<float> temp9(N, 0.0f);
				   thrust::host_vector<float> temp10(N, 0.0f);
				   
				   // device side vectors
				   thrust::device_vector<float> d_Xt = Xt;
				   thrust::device_vector<float> d_Xp = Xp;
				   thrust::device_vector<float> d_DXp = DXp;
				   thrust::device_vector<float> d_DW = DW;
				   thrust::device_vector<float> d_DX = DX;
				   thrust::device_vector<float> d_DC1 = DC1;
				   thrust::device_vector<float> d_DC2 = DC2;
				   thrust::device_vector<float> d_DC3 = DC3;
				   thrust::device_vector<float> d_Rp = Rp;
				   thrust::device_vector<float> d_Rt = Rt;
				   thrust::device_vector<float> d_Rs = Rs;
				   thrust::device_vector<float> d_Vt = Vt;
				   thrust::device_vector<float> d_Pt = Pt;				   
				   thrust::device_vector<float> d_Gt = Gt;				   
				   thrust::device_vector<float> d_R1 = R1;
				   thrust::device_vector<float> d_R2 = R2;
				   thrust::device_vector<float> d_R3 = R3;
				   
				   thrust::device_vector<float> d_temp1 = temp1;
				   thrust::device_vector<float> d_temp2 = temp2;
				   thrust::device_vector<float> d_temp3 = temp3;
				   thrust::device_vector<float> d_temp4 = temp4;
				   thrust::device_vector<float> d_temp5 = temp5;
				   thrust::device_vector<float> d_temp6 = temp6;
				   thrust::device_vector<float> d_temp7 = temp7;
				   thrust::device_vector<float> d_temp8 = temp8;
				   thrust::device_vector<float> d_temp9 = temp9;
				   thrust::device_vector<float> d_temp10 = temp10;
				   thrust::device_vector<float> d_result = result;				   
				    
				   float alpha = 1.0f;
				   float beta = 0.0f;
				   
				   float W_ini = 0.3f;
				   float W_min = 0.1f;
				   float F_max = 10.0f;		
				   int C1 = 2;
				   int C2 = 2;
				   float W = 5.0f;
				   float Rk = 4.5f;				

				 	 
   			       // step 1: calculate learning factors
				   // 1.1 fitness func values
				   thrust::host_vector<float> fitnessValVector = EvaluateFitnessFunctionVector(N, Rk, Xp, Xt);
				   
				   // 1.2 fit1
				   thrust::copy(fitnessValVector.begin(), fitnessValVector.end(), d_temp1.begin());
				   
				   // 1.3 fit2
				   thrust::device_vector<float>::iterator iter=thrust::max_element(d_temp1.begin(),d_temp1.end());
				   unsigned int position = iter - d_temp1.begin();
				   float maxFitnessVal = d_temp1[position];
				   thrust::fill(d_temp2.begin(), d_temp2.end(), maxFitnessVal);
				   
				   // 1.4 fit3
				   thrust::host_vector<float> randomParticleVector = EvaluateFitnessFunctionVector(N, Rk, Rp, Rs);
				   thrust::fill(d_temp3.begin(), d_temp3.end(), randomParticleVector[0]);
				   //thrust::copy(randomParticleVector.begin(), randomParticleVector.end(), d_temp3.begin());
				   
				   // 1.5 fit1 + fit2
				   thrust::copy(d_temp2.begin(), d_temp2.end(), temp4.begin());
				   d_temp4 = temp4;
				   CUBLAS_SAFE_CALL(cublasSaxpy(handle, N, &alpha, thrust::raw_pointer_cast(d_temp1.data()), 1,
									thrust::raw_pointer_cast(d_temp4.data()), 1));
				   
				   
				   
				   // 1.6 D = fit1 + fit2 + fit3
				   CUBLAS_SAFE_CALL(cublasSaxpy(handle, N, &alpha, thrust::raw_pointer_cast(d_temp3.data()), 1,
									thrust::raw_pointer_cast(d_temp4.data()), 1));
				   // 1.7 c1 = 2.8*fit1/(D)
				   alpha = 2.8;
				   CUBLAS_SAFE_CALL(cublasSscal(handle, N, &alpha, thrust::raw_pointer_cast(d_temp1.data()), 1));
				   thrust::host_vector<float> a = d_temp1;
				   thrust::host_vector<float> b= d_temp4;
				   auto c1 = Divide( a, b ) ;				   
				   
				   // 1.8 c2 = 2.8*fit2/(D)
				   CUBLAS_SAFE_CALL(cublasSscal(handle, N, &alpha, thrust::raw_pointer_cast(d_temp2.data()), 1));
				   thrust::host_vector<float> g = d_temp2;
				   auto c2 = Divide( g, b ) ;				   
				   
				   // 1.9 c3 = 2.8*fit3/(D)
				   CUBLAS_SAFE_CALL(cublasSscal(handle, N, &alpha, thrust::raw_pointer_cast(d_temp3.data()), 1));
				   thrust::host_vector<float> h = d_temp3;
				   auto c3 = Divide( h, b );
				   
				   // step 2:: velocity update
				   // 2.1:: A = w*Vt  - get this using (Diag(W))*Vt	, result stored in d_temp5		   
				   // setup diagonal matrix
				   for(int i=0; i<N; i++) {
					   DW[Idx2C(i,i,N)] = weightVec[i];
				   }	
				   
				   // calc as 1*(Diag(W))*Vt
				   alpha = 1.f;
				   thrust::copy(DW.begin(), DW.end(), d_DW.begin());
				   CUBLAS_SAFE_CALL(cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha, 
				                    thrust::raw_pointer_cast(d_DW.data()), N, thrust::raw_pointer_cast(d_Vt.data()), 1, &beta, thrust::raw_pointer_cast(d_temp5.data()), 1));			   
				   
				   
				   // 2.2:: calc B = c1*r1*(Pbest - Xt), result stored in d_temp8
				   // 2.2.1 X = c1*r1
				   float* randomArray = GetRandomNumArray(N);
				   memcpy(thrust::raw_pointer_cast(R1.data()), &randomArray[0], N);
				   thrust::copy(R1.begin(), R1.end(), d_R1.begin());
				   
				   // c1*r1  - get this using (Diag(c1))*r1			   
				   // setup diagonal matrix
				   for(int i=0; i<N; i++) {
					   DC1[Idx2C(i,i,N)] = c1[i];
				   }	
				   
				   // calc as 1*(Diag(c1))*r1
				   alpha = 1.f;
				   beta = 0.f;
				   thrust::copy(DC1.begin(), DC1.end(), d_DC1.begin());
				   CUBLAS_SAFE_CALL(cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha, 
				                    thrust::raw_pointer_cast(d_DC1.data()), N, thrust::raw_pointer_cast(d_R1.data()), 1, &beta, thrust::raw_pointer_cast(d_temp6.data()), 1));
				   
				   
				   
				   // 2.2.2 Y = (Pbest - Xt)
				   alpha = -1.f;
				   thrust::copy(Pt.begin(), Pt.end(), d_temp7.begin());
				   CUBLAS_SAFE_CALL(cublasSaxpy(handle, N, &alpha, thrust::raw_pointer_cast(d_Xt.data()), 1,
									thrust::raw_pointer_cast(d_temp7.data()), 1));
				   
				   
				   // 2.2.3 X*Y - get this using (Diag(X))*Y			   
				   // setup diagonal matrix
				   for(int i=0; i<N; i++) {
					   DX[Idx2C(i,i,N)] = temp6[i];
				   }	
				   
				   // calc 1*(Diag(X))*Y
				   alpha = 1.f;
				   beta = 0.f;
				   CUBLAS_SAFE_CALL(cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha, 
				                    thrust::raw_pointer_cast(d_DX.data()), N, thrust::raw_pointer_cast(d_temp7.data()), 1, &beta, thrust::raw_pointer_cast(d_temp8.data()), 1));				   
				   
				   
				   // step 2.3:: C = c2*r2*(Gbest - Xt), result stored in d_temp9
				   // 2.3.1 X = c2*r2
				   randomArray = GetRandomNumArray(N);
				   memcpy(thrust::raw_pointer_cast(R2.data()), &randomArray[0], N);
				   thrust::copy(R2.begin(), R2.end(), d_R2.begin());
				   
				   // c2*r2  - get this using (Diag(c2))*r2			   
				   // setup diagonal matrix
				   for(int i=0; i<N; i++) {
					   DC2[Idx2C(i,i,N)] = c2[i];
				   }	
				   
				   // calc as 1*(Diag(c2))*r2
				   alpha = 1.f;
				   beta = 0.f;
				   thrust::copy(temp6.begin(), temp6.end(), d_temp6.begin());	
                   CUBLAS_SAFE_CALL(cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha, 
				                    thrust::raw_pointer_cast(d_DC2.data()), N, thrust::raw_pointer_cast(d_R2.data()), 1, &beta, thrust::raw_pointer_cast(d_temp6.data()), 1));
				   
				   // 2.3.2 Y = (Gbest - Xt)
				   alpha = -1.f;
				   thrust::copy(Gt.begin(), Gt.end(), d_temp7.begin());
				   CUBLAS_SAFE_CALL(cublasSaxpy(handle, N, &alpha, thrust::raw_pointer_cast(d_Xt.data()), 1,
									thrust::raw_pointer_cast(d_temp7.data()), 1));
				   
				   
				   // 2.3.3 X*Y - get this using (Diag(X))*Y			   
				   // setup diagonal matrix
				   thrust::copy(d_temp6.begin(), d_temp6.end(), temp6.begin());
				   for(int i=0; i<N; i++) {
					   DX[Idx2C(i,i,N)] = temp6[i];
				   }	
				   
				   // calc 1*(Diag(X))*Y
				   alpha = 1.f;
				   beta = 0.f;
				   thrust::copy(DX.begin(), DX.end(), d_DX.begin());
				   CUBLAS_SAFE_CALL(cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha, 
				                    thrust::raw_pointer_cast(d_DX.data()), N, thrust::raw_pointer_cast(d_temp7.data()), 1, &beta, thrust::raw_pointer_cast(d_temp9.data()), 1));
				   
				   
				   // step 2.4:: D = c3*r3*(Rbest - Xt, result stored in d_temp10
				   // 2.3.1 X = c3*r3
				   randomArray = GetRandomNumArray(N);
				   memcpy(thrust::raw_pointer_cast(R3.data()), &randomArray[0], N);
				   thrust::copy(R3.begin(), R3.end(), d_R3.begin());
				   
				   // c3*r3  - get this using (Diag(c3))*r3			   
				   // setup diagonal matrix
				   for(int i=0; i<N; i++) {
					   DC3[Idx2C(i,i,N)] = c3[i];
				   }	
				   
				   // calc as 1*(Diag(c3))*r3
				   alpha = 1.f;
				   beta = 0.f;
				   thrust::copy(DC3.begin(), DC3.end(), d_DC3.begin());
				   CUBLAS_SAFE_CALL(cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha, 
				                    thrust::raw_pointer_cast(d_DC3.data()), N, thrust::raw_pointer_cast(d_R3.data()), 1, &beta, thrust::raw_pointer_cast(d_temp6.data()), 1));
				   
				   // 2.3.2 Y = (Rbest - Xt)
				   alpha = -1.f;
				   thrust::copy(Rt.begin(), Rt.end(), d_temp7.begin());
				   CUBLAS_SAFE_CALL(cublasSaxpy(handle, N, &alpha, thrust::raw_pointer_cast(d_Xt.data()), 1,
									thrust::raw_pointer_cast(d_temp7.data()), 1));
				   
				   // 2.3.3 X*Y - get this using (Diag(X))*Y			   
				   // setup diagonal matrix
				   for(int i=0; i<N; i++) {
					   DX[Idx2C(i,i,N)] = temp6[i];
				   }	
				   
				   // calc 1*(Diag(X))*Y
				   alpha = 1.f;
				   beta = 0.f;
				   thrust::copy(DX.begin(), DX.end(), d_DX.begin());
				   CUBLAS_SAFE_CALL(cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha, 
				                    thrust::raw_pointer_cast(d_DX.data()), N, thrust::raw_pointer_cast(d_temp7.data()), 1, &beta, thrust::raw_pointer_cast(d_temp10.data()), 1));
				   
				   
				   // step 2.5 K = A + B + C + D
				   // 2.5.1 p = A + B
				   CUBLAS_SAFE_CALL(cublasSaxpy(handle, N, &alpha, thrust::raw_pointer_cast(d_temp5.data()), 1,
									thrust::raw_pointer_cast(d_temp8.data()), 1));
				   
				   // 2.5.1 q = C + D
				   CUBLAS_SAFE_CALL(cublasSaxpy(handle, N, &alpha, thrust::raw_pointer_cast(d_temp9.data()), 1,
									thrust::raw_pointer_cast(d_temp10.data()), 1));
				   
				   // 2.5.1 p + q
				   CUBLAS_SAFE_CALL(cublasSaxpy(handle, N, &alpha, thrust::raw_pointer_cast(d_temp8.data()), 1,
									thrust::raw_pointer_cast(d_temp10.data()), 1));
				   
				   // copy back result
				   thrust::copy(d_temp10.begin(), d_temp10.end(), result.begin());
				   
				}
				catch(std::exception &e) {
					eptr = std::current_exception();
				}
				catch(...) {
					eptr = std::current_exception();
				}
				
				//cleanup
				// thrust vectors cleaned up automatically
				CUBLAS_SAFE_CALL(cublasDestroy(handle));
				
				if(eptr)
				{
					std::rethrow_exception(eptr);
				}
				
				return result;	
		}
		
		
		void UpdatePosition(int N, thrust::host_vector<float> velocityVector){
			
			float alpha = 1.0f;
			
			std::exception_ptr eptr;
			cublasHandle_t handle;
			
			
			try {
				   // cuBLAS handle creation
				   CUBLAS_SAFE_CALL(cublasCreate(&handle));
				   
				   thrust::host_vector<float> Xt(N, 2.0f);
				   thrust::host_vector<float> temp1(N, 0.0f);
				   
				   // calculate Xt + Vt : result stored in temp1
				   memcpy(thrust::raw_pointer_cast(temp1.data()), thrust::raw_pointer_cast(velocityVector.data()), N);
				   thrust::device_vector<float> d_temp1 = temp1;
				   thrust::device_vector<float> d_Xt = Xt;
			
				   CUBLAS_SAFE_CALL(cublasSaxpy(handle, N, &alpha, thrust::raw_pointer_cast(d_Xt.data()), 1,
									thrust::raw_pointer_cast(d_temp1.data()), 1));
				   
			}
			catch(std::exception &e) {
				eptr = std::current_exception();
			}
			catch(...) {
				eptr = std::current_exception();
			}
			
			//cleanup
			// thrust vectors cleaned up automatically
			CUBLAS_SAFE_CALL(cublasDestroy(handle));			
						
			if(eptr)
			{
				std::rethrow_exception(eptr);
			}	
			
		};

		
		void PFTimeUpdate(int p, int N, int T, int randomNumComplexityConst) {
		
			std::exception_ptr eptr;
			cublasHandle_t handle;
			cusolverDnHandle_t cusolver_handle;			
							
		    try {
			
				int matrixQRows = p;
				int matrixQCols = p;
				int matrixT1Rows = p;
				int matrixT1Cols = p; 
				int matrixT2Rows = p;
				int matrixT2Cols = N;
				int matrixApRows = p;
				int matrixApCols = p;
				int matrixXpRows = p;
				int matrixXpCols = N;		
				int matrixWRows = p;
				int matrixWCols = N;
				int matrixMRows = p;
				int matrixMCols = p;
				int identityMatrixRows = N;
										
				// host side matrices
				thrust::host_vector<float> Q(matrixQRows*matrixQCols);		
				thrust::host_vector<float> T1(matrixT1Rows*matrixT1Cols);		
				thrust::host_vector<float> T2(matrixT2Rows*matrixT2Cols);		
				thrust::host_vector<float> Ap(matrixApRows*matrixApCols);
				thrust::host_vector<float> Xp(matrixXpRows*matrixXpCols);			
				thrust::host_vector<float> W(matrixWRows*matrixWCols);
				thrust::host_vector<float> M(matrixMRows*matrixMCols);
				thrust::host_vector<float> Identity(identityMatrixRows*identityMatrixRows, 0.f);
				
				thrust::host_vector<float> temp2(matrixT2Rows, matrixT2Cols);	
				thrust::host_vector<float> temp3(matrixXpRows, matrixXpCols);	
				thrust::host_vector<float> temp4(matrixXpRows, matrixXpCols);	
				
				Identity = GetIdentityMatrix(identityMatrixRows);
				
				// Copy inputs to device
				thrust::device_vector<float> d_ap = Ap;
				thrust::device_vector<float> d_q = Q;
				thrust::device_vector<float> d_t1 = T1;
				thrust::device_vector<float> d_t2 = T2;
				thrust::device_vector<float> d_xp = Xp;			
				thrust::device_vector<float> d_w = W;	
				thrust::device_vector<float> d_identity = Identity;	
				
				thrust::device_vector<float> d_temp2 = temp2;
				thrust::device_vector<float> d_temp3 = temp3;
				thrust::device_vector<float> d_temp4 = temp4;
							
				// cuBLAS handle creation
				CUBLAS_SAFE_CALL(cublasCreate(&handle));
				CUSOLVER_SAFE_CALL(cusolverDnCreate(&cusolver_handle));
				
				float alpha = 1.f;
				float beta  = 0.f;	
				
				// step 1 cholesky factorisation of M
				thrust::host_vector<float> matrixM = GetSymmetricPostiveDefiniteMatrix(matrixMRows);
				thrust::device_vector<float> d_M = matrixM;
				float* devM = thrust::raw_pointer_cast(d_M.data());
				int lWork =0;
				int* devInfo =0;
				float* workspace =0;
				 
				CUSOLVER_SAFE_CALL(cusolverDnSpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_UPPER, matrixMRows, devM, matrixMRows, &lWork));
				
				CUDA_SAFE_CALL(cudaMalloc((void **)&workspace, sizeof(float)*(lWork)));
				CUDA_SAFE_CALL(cudaMalloc((void **)&devInfo , sizeof (int)));
				
				CUSOLVER_SAFE_CALL(cusolverDnSpotrf(cusolver_handle, CUBLAS_FILL_MODE_UPPER, matrixMRows, devM, matrixMRows, workspace, lWork, devInfo));
				CUDA_SAFE_CALL(cudaDeviceSynchronize());
				
				cudaFree(workspace);
				cudaFree(devInfo);
				
				// step 2 - perform T_1*T_2 (resultant matrix is p x N)
				beta  = 0.f;						
				CUBLAS_SAFE_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrixT2Rows, matrixT2Cols, 
								matrixT1Cols, &alpha, thrust::raw_pointer_cast(d_t1.data()), matrixT1Rows, thrust::raw_pointer_cast(d_t2.data()), matrixT2Rows, &beta, thrust::raw_pointer_cast(d_temp2.data()), matrixT2Rows));			
				CUDA_SAFE_CALL(cudaDeviceSynchronize());

				// step 3 - perform T3 = A_p*X_p (resultant matrix is p x N)
				CUBLAS_SAFE_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrixXpRows, matrixXpCols, 
								matrixApCols, &alpha, thrust::raw_pointer_cast(d_ap.data()), matrixApRows, thrust::raw_pointer_cast(d_xp.data()), matrixXpRows, &beta, thrust::raw_pointer_cast(d_temp3.data()), matrixXpRows));			
				CUDA_SAFE_CALL(cudaDeviceSynchronize());
				
				// step 4 - perform T3 + T4 + W
				beta  = 1.f;
				// done in two steps
				// (a) temp = 1*T3*Id + 1*T4
				CUBLAS_SAFE_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrixXpRows, matrixXpCols, 
								matrixXpCols, &alpha, thrust::raw_pointer_cast(d_temp3.data()), matrixXpRows, thrust::raw_pointer_cast(d_identity.data()), matrixXpCols, &beta, thrust::raw_pointer_cast(d_temp4.data()), matrixXpRows));			
				CUDA_SAFE_CALL(cudaDeviceSynchronize());
				
				
				// (b) final = 1*temp*Id + 1*W
				CUBLAS_SAFE_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrixXpRows, identityMatrixRows, 
								matrixXpCols, &alpha, thrust::raw_pointer_cast(d_temp4.data()), matrixXpRows, thrust::raw_pointer_cast(d_identity.data()), identityMatrixRows, &beta, thrust::raw_pointer_cast(d_w.data()), matrixWRows));			
				CUDA_SAFE_CALL(cudaDeviceSynchronize());
							
				
				// do last
				// copy result to host
				//thrust::copy(d_w.begin(), d_w.end(), res.begin());
				
				
			} catch (std::exception &e) {
				eptr = std::current_exception();
				printf("PFTimeUpdate::Error::%s \n", e.what());
			}
			catch (...) {
			  eptr = std::current_exception();
			  printf("PFTimeUpdate::Unknown Error \n");
			}			
				
			// cleanup
			// thrust vectors are automatically deleted on return
			// free up handles
			CUBLAS_SAFE_CALL(cublasDestroy(handle));
			CUSOLVER_SAFE_CALL(cusolverDnDestroy(cusolver_handle));			
			
			if(eptr) {
			   std::rethrow_exception(eptr);
			}		
		};
		
		thrust::host_vector<float> EvaluateFitnessFunctionVector(int N, float Rk, thrust::host_vector<float> Y_new, 
																 thrust::host_vector<float> Y_pred)
		{
			std::exception_ptr eptr;
			cublasHandle_t handle;
			thrust::host_vector<float> result(N, 0.0f);

			try {
				   CUBLAS_SAFE_CALL(cublasCreate(&handle));
				   
				   // host side vectors
				   thrust::host_vector<float> DXp(N*N, 0.0f);		
				   thrust::host_vector<float> temp4(N, 0.0f);		
				   thrust::host_vector<float> temp5(N, 0.0f);
				   thrust::host_vector<float> temp6(N, 0.0f);
				   
				   // Copy to device side
				   thrust::device_vector<float> d_DXp = DXp;
				   thrust::device_vector<float> d_temp4 = temp4;
				   thrust::device_vector<float> d_temp5 = temp5;
				   thrust::device_vector<float> d_temp6 = temp6;
				   thrust::device_vector<float> d_Ynew = Y_new;
				   thrust::device_vector<float> d_Ypred = Y_pred;				   
					
				   float alpha = -1.0f;
				   float beta = 0.0f;
				   
				   // step 1: evaluate fitness function 
				   // step 3.1 Y_diff = Y_new - Y_pred
				   // copy input data from host to device
				   thrust::copy(Y_new.begin(), Y_new.end(), d_temp4.begin());
				   CUBLAS_SAFE_CALL(cublasSaxpy(handle, N, &alpha, thrust::raw_pointer_cast(d_Ypred.data()), 1, 
								    thrust::raw_pointer_cast(d_temp4.data()), 1));
				   
				   // step 3.2 (y_new - y_pred)^2 - get this using (Diag(Y_diff))*Y_diff			   
				   // setup diagonal matrix
				   for(int i=0; i<N; i++) {
					   DXp[Idx2C(i,i,N)] = temp4[i];
				   }	
				   
				   // calc B = (-0.5/Rk)*(Diag(Y_diff))*Y_diff
				   alpha = (-0.5f/Rk);
				   CUBLAS_SAFE_CALL(cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha, thrust::raw_pointer_cast(d_DXp.data()), N, thrust::raw_pointer_cast(d_temp4.data()), 1, &beta, thrust::raw_pointer_cast(d_temp5.data()), 1));
				   
				   // calc exp(b_i) 				   
				   // call the `float' version of exp func
				   thrust::transform(d_temp5.begin(), d_temp5.end(), d_temp6.begin(), expf_func());				   
				   
				   // copy back result to host
				   result = d_temp6;	

			}
			catch(std::exception &e) {
				eptr = std::current_exception();
			}
			catch(...) {
				eptr = std::current_exception();
			}
			
			//cleanup
			// thrust vectors auto cleanup
			CUBLAS_SAFE_CALL(cublasDestroy(handle));
			
			if(eptr)
			{
				std::rethrow_exception(eptr);
			}
			
			return result;
		};

		
		
		/*
		Method to invert a square matrix with non zero determinant.

		Input Parameters:
		  x - An rows x rows matrix.
		  rows - number of rows/columns of matrix.  
			
		Output Parameters:
		   x - the inverted matrix
		*/
		float* InvertMatrix(thrust::host_vector<float> x, int rows) {
			cublasHandle_t handle;
			CUBLAS_SAFE_CALL(cublasCreate(&handle));

			float** al;
			float** ac;
			float* d_l;
			float* d_c;
			int* d_pivots;
			int* d_info;
			float* inv;
			
			// storage on host for result
			thrust::host_vector<float> inverse(rows * rows);

			size_t size = rows * rows * sizeof(float);			
			// int n = rows * rows;
			
			CUDA_SAFE_CALL(cudaMalloc(&al, sizeof(float*)));
			CUDA_SAFE_CALL(cudaMalloc(&ac, sizeof(float*)));
			CUDA_SAFE_CALL(cudaMalloc(&d_l, size));
			CUDA_SAFE_CALL(cudaMalloc(&d_c, size));
			CUDA_SAFE_CALL(cudaMalloc(&d_pivots, rows * sizeof(int)));
			CUDA_SAFE_CALL(cudaMalloc(&d_info, sizeof(int)));

			CUDA_SAFE_CALL(cudaMemcpy(d_l, thrust::raw_pointer_cast(x.data()), size, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(al, &d_l, sizeof(float*), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(ac, &d_c, sizeof(float*), cudaMemcpyHostToDevice));			

			CUBLAS_SAFE_CALL(cublasSgetrfBatched(handle, rows, al, rows, d_pivots, d_info, 1));
			CUDA_SAFE_CALL(cudaDeviceSynchronize());

			CUBLAS_SAFE_CALL(cublasSgetriBatched(handle, rows, (const float **)al, rows, d_pivots, ac, rows, d_info, 1));
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			
			// copy result to host
			inv = (float*)malloc(size);			
			CUDA_SAFE_CALL(cudaMemcpy(inv, d_c, size, cudaMemcpyDeviceToHost));
			//thrust::copy(inv, (inv + rows*rows), inverse.begin());
			
			// cleanup
			CUDA_SAFE_CALL(cudaFree(al));
			CUDA_SAFE_CALL(cudaFree(ac));
			CUDA_SAFE_CALL(cudaFree(d_l));
			CUDA_SAFE_CALL(cudaFree(d_c));
			CUDA_SAFE_CALL(cudaFree(d_pivots));
			CUDA_SAFE_CALL(cudaFree(d_info));

			CUBLAS_SAFE_CALL(cublasDestroy(handle));

			return inv;
		};
		
		thrust::host_vector<float> GetIdentityMatrix(int identityMatrixRows){
		
			thrust::host_vector<float> identity(identityMatrixRows*identityMatrixRows, 0.f);
				
			for (int i=0; i <= identityMatrixRows; i++){
			   for (int j=0; j<=identityMatrixRows; j++) {
				 if (i==j) {
				   identity[i] = 1;
				 }
			   }
			}
			
			return identity;
		
		};
		
		thrust::host_vector<float> GetSymmetricPostiveDefiniteMatrix(int n){
		
			// cuBLAS handle creation
			cublasHandle_t cublas_handle;
			CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle));
			
			thrust::host_vector<float> L(n*n, 0.f);
			thrust::host_vector<float> C(n*n, 0.f);
			
			int size = sizeof(float)*n*n;
			
			int i,j;
			float* data = thrust::raw_pointer_cast(L.data());
			for (j = 0; j < n; j++){
				for (i = 0; i < n; i++){
					if(j <= i) {
						data[Idx2C(i,j,n)]= 4;
					}
				}
			}
			
			thrust::device_vector<float> d_l = L;
			thrust::device_vector<float> d_c = C;
			
			float alpha = 1.0;
			float beta = 0.0;

			// compute matrix C = L<L^T>
			CUBLAS_SAFE_CALL(cublasSsyrk(cublas_handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, n, &alpha, thrust::raw_pointer_cast(d_l.data()), n, &beta, 
						     thrust::raw_pointer_cast(d_c.data()), n));
						   
			// copy result
			thrust::copy(d_l.begin(), d_l.end(), C.begin());	
			
			CUBLAS_SAFE_CALL(cublasDestroy(cublas_handle));
			
			L.clear();
			d_l.clear();
			d_c.clear();
			
			//return thrust::raw_pointer_cast(C.data());

			return C;
			
		};
		
		thrust::host_vector<float> Divide( thrust::host_vector<float> a, thrust::host_vector<float> b)
		{
			thrust::host_vector<float> result(N_loc, 0.0f);
			
			// device side vectors
			thrust::device_vector<float> d_a = a;
			thrust::device_vector<float> d_b = b;
			thrust::device_vector<float> d_res = result;
			
			thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_res.begin(), thrust::divides<float>());
			
			// copy back result
			thrust::copy(d_res.begin(), d_res.end(), result.begin());
			return result ;
		};
		
		float* GetRandomNumArray(int size) {
		
			size_t n = size;
			size_t i;
			curandGenerator_t gen;
			float* devData;
			float* hostData;

			/* Allocate n floats on host */
			hostData = (float *)calloc(n, sizeof(float));

			/* Allocate n floats on device */
			CUDA_SAFE_CALL(cudaMalloc((void **)&devData, n*sizeof(float)));

			/* Create pseudo-random number generator */
			curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW); // CURAND_SAFE_CALL
						
			/* Set seed */
			curandSetPseudoRandomGeneratorSeed(gen, 1234);

			/* Generate n floats on device */
			curandGenerateUniform(gen, devData, n);

			/* Copy device memory to host */
			CUDA_SAFE_CALL(cudaMemcpy(hostData, devData, n * sizeof(float), cudaMemcpyDeviceToHost));
				
			/* Cleanup */
			curandDestroyGenerator(gen);
			CUDA_SAFE_CALL(cudaFree(devData));
			
			return hostData;		
		}
		
		private:
		     int p_loc;
			 int N_loc;
			 int T_loc;
			 int randomNumComplexityConst_loc;
			 
};

}


int run_on_measurements_cuda(int p, int N, int T, int randomNumComplexityConst, int maxThreads){
  
  RunAPSO *psopf = new RunAPSO(p, N, T, randomNumComplexityConst);
  psopf->RunOnMeasurements();
  return 1;
}





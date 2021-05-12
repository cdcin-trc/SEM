#include "adaptiveswarmoptimisedpfworker.h"
#include <core/component.h>
#include <pthread.h>
#include <math.h>
#include "apso_cuda.h"
#include <chrono>
#include <iostream>
#include <assert.h>
#include <thread>         
#include <mutex>  
#include <random>  
#include "cblas.h"
#include "lapack.h"
#include <vector>
#include <algorithm>
#include <functional>
#include <iterator>  

std::mutex mtx;

/*
This MEDEA worker emulates the workload associated with the Adaptive Particle Swarm Optimised (APSO) Particle Filter approach.
*/

// Converts 2D matrix indices to CblasColMajor storage format (linear) index
#define Idx2C(i,j,ld) ((j)*(ld) + (i))

// Rounds a float/double value to 2 decimal places
#define nearest(x) round(x * 100) / 100

using namespace std;

AdaptiveSwarmOptimisedPFWorker::AdaptiveSwarmOptimisedPFWorker(const BehaviourContainer& container, const std::string& inst_name) : Worker(container, GET_FUNC, inst_name){
	cpu_worker_ = std::unique_ptr<Cpu_Worker>(new Cpu_Worker(*this, "cpu_worker"));    
}

AdaptiveSwarmOptimisedPFWorker::~AdaptiveSwarmOptimisedPFWorker(){
	cpu_worker_.reset();	
}

/*
Sets the APSO particle filter properties.
Input Parameters:
  standardPF_P - The state dimension from the standard particle filter.
  numParticles_N - The number of particles in the system.
  numIterations_T - The number of iteration steps in PSO-based sampling.
  randomNumComplexityConstant -  estimated random number calculation coefficient.
  maxThreads - The maximum number of threads to utilised when performing calculation (applies to CPU only).
  
Output Parameters:
   None
*/

void AdaptiveSwarmOptimisedPFWorker::SetProperties (int standardPF_P, int numParticles_N, int numIterations_T, int randomNumComplexityConstant, int maxThreadsAllowable) {
	
	auto work_id = get_new_work_id();
    //Log(GET_FUNC, Logger::WorkloadEvent::STARTED, work_id);
   
	try {
		
		mtx.lock();
		p = standardPF_P;
		N = numParticles_N;
		T = numIterations_T;
		randomNumComplexityConst = randomNumComplexityConstant;
		maxThreads = maxThreadsAllowable;
	
	} catch (std::exception &e) {
		const std::string & message = get_arg_string_variadic("Exception: \"%s\"", e.what());
		Log(GET_FUNC, Logger::WorkloadEvent::ERROR, work_id, message);
	} catch (...) {
		Log(GET_FUNC, Logger::WorkloadEvent::ERROR, work_id, "Unrecognized exception.");
	}
	
	//Log(GET_FUNC, Logger::WorkloadEvent::FINISHED, work_id);	
	
	// free locks
	mtx.unlock();
}

/*
Executes the PSO particle filter based on flops.
Input Parameters:
	None  
Output Parameters:
   None
*/

void AdaptiveSwarmOptimisedPFWorker::RunOnMeasurementsFlops () {
	
	auto work_id = get_new_work_id();
    //Log(GET_FUNC, Logger::WorkloadEvent::STARTED, work_id);
   
	try {
		// Record start time
	   auto start = std::chrono::high_resolution_clock::now();
	   mtx.lock();
	   
	   double calc_adaptive_weights = 15*N*T;
	   double update_velocity = (25*N + 15*N*N)*T;
	   double update_position = 7*N*T;
	   double pf_time_update = p*p*p/3 + 2*p*p + 4*p*p*N + p*N*randomNumComplexityConst  - p*N;
	   
	   double m_targetoperationsmillions = abs((calc_adaptive_weights + 
												update_velocity + 
												pf_time_update + 
												update_position)/1000000);
	   // printf("m_targetoperationsmillions:: %0.2f \n",m_targetoperationsmillions);
			  
	   if (maxThreads>1){
		   int numberOfThreads = std::thread::hardware_concurrency();
		   // Cap max threads to those available
		   if (numberOfThreads>maxThreads){
			   numberOfThreads = maxThreads;
		   }	   
		   AdaptiveSwarmOptimisedPFWorker::cpu_worker_->MWIP(m_targetoperationsmillions, numberOfThreads, true);
	   }
	   else{
		  // single-threaded operation
		  AdaptiveSwarmOptimisedPFWorker::cpu_worker_->MWIP(m_targetoperationsmillions);	   
	   }   
	   
	   // Record end time
	   // auto finish = std::chrono::high_resolution_clock::now();
	   // std::chrono::duration<double> elapsed = finish - start;
	   // std::cout << "Elapsed time(RunOnMeasurementsFlops, ops-only): " << elapsed.count() << " s\n";
				
	} catch (std::exception &e) {
		const std::string & message = get_arg_string_variadic("Exception: \"%s\"", e.what());
		Log(GET_FUNC, Logger::WorkloadEvent::ERROR, work_id, message);
	} catch (...) {
		Log(GET_FUNC, Logger::WorkloadEvent::ERROR, work_id, "Unrecognized exception.");
	}
	
	mtx.unlock();
	//Log(GET_FUNC, Logger::WorkloadEvent::FINISHED, work_id);
}

/*
Executes the PSO particle filter based on actual algorithms.
Input Parameters:
	None  
Output Parameters:
   None
*/
void AdaptiveSwarmOptimisedPFWorker::RunOnMeasurements () {
	
	auto work_id = get_new_work_id();
    //Log(GET_FUNC, Logger::WorkloadEvent::STARTED, work_id);
   
	try {
	   	   
	   // Record start time
	   auto start = std::chrono::high_resolution_clock::now();
	   mtx.lock();
	   	   
	   // Call GPU code.. will return immediately if GPUs are not supported
	   int res = run_on_measurements_cuda(p, N, T, randomNumComplexityConst, maxThreads);
	   
	   if(res == 0) {		   
		    
			// Call CPU code
			//printf("GPU not supported...running on CPUs \n");
			for(int t=0; t<T; t++) {
				float* weightVec = CalculateAdaptiveWeights(N, T, t);
				float* velocityVec = UpdateVelocity(N, weightVec);
			    UpdatePosition(N, velocityVec);
				if(weightVec) free(weightVec);
				if(velocityVec) free(velocityVec);				
			}	

			PFTimeUpdate(p, N, randomNumComplexityConst);							
		}

  	   // Record end time
	   //auto finish = std::chrono::high_resolution_clock::now();
	   //std::chrono::duration<double> elapsed = finish - start;
	   //std::cout << elapsed.count() << "\n";
	   //std::cout << "Elapsed time(AdaptiveSwarmOptimisedPFWorker::RunOnMeasurements, actual): " << elapsed.count() << " s\n";
				
	} catch (std::exception &e) {
		const std::string & message = get_arg_string_variadic("Exception: \"%s\"", e.what());
		Log(GET_FUNC, Logger::WorkloadEvent::ERROR, work_id, message);
	} catch (...) {
		Log(GET_FUNC, Logger::WorkloadEvent::ERROR, work_id, "Unrecognized exception.");
	}
	
	// freeup resources
	mtx.unlock();
	//Log(GET_FUNC, Logger::WorkloadEvent::FINISHED, work_id);	
}
	
/*
Method to execute the worker's suite of unit tests (CPU only).
*/
void AdaptiveSwarmOptimisedPFWorker::RunUnitTests() 
{
	auto work_id = get_new_work_id();
    //Log(GET_FUNC, Logger::WorkloadEvent::STARTED, work_id);
   
	try {
	   
				
	} catch (std::exception &e) {
		const std::string & message = get_arg_string_variadic("Exception: \"%s\"", e.what());
		Log(GET_FUNC, Logger::WorkloadEvent::ERROR, work_id, message);
	} catch (...) {
		Log(GET_FUNC, Logger::WorkloadEvent::ERROR, work_id, "Unrecognized exception.");
	}
	
	//Log(GET_FUNC, Logger::WorkloadEvent::FINISHED, work_id); 
	
};

// Private members
// CPU side code
// check for memory leaks - DONE
float* AdaptiveSwarmOptimisedPFWorker::CalculateAdaptiveWeights(int N, int T, int step){
	str:exception_ptr eptr;
	float* Xt = 0;
	float* Xp = 0;
	float* DXp = 0;
	
	float* temp1 = 0;
	float* temp2 = 0;
	float* temp3 = 0;	
	float* temp4 = 0;	
	float* temp5 = 0;
	float* temp6 = 0;
	float* result = 0;
	
	int C1 = 2;
	int C2 = 2;
	float W = 5.0f;
	float Rk = 4.5f;
	
	//auto start = std::chrono::high_resolution_clock::now();
	   
	
	try {
		   Xt = (float*)malloc(N*sizeof(float));
		   Xp = (float*)malloc(N*sizeof(float));
		   DXp = (float*)malloc(N*N*sizeof(float));
		   
		   temp1 = (float*)malloc(N*sizeof(float));
		   temp2 = (float*)malloc(N*sizeof(float));
		   temp3 = (float*)malloc(N*sizeof(float));
		   temp4 = (float*)malloc(N*sizeof(float));
		   temp5 = (float*)malloc(N*sizeof(float));
		   temp6 = (float*)malloc(N*sizeof(float));
		   result = (float*)malloc(N*sizeof(float));
		   
		   std::fill_n(Xt, N, 2.0f);
		   std::fill_n(Xp, N, 1.0f);
		   std::fill_n(DXp, N*N, 0.0f);
		   std::fill_n(result, N, 0.0f);				   

		   float alpha = -1.0f;
		   float beta = 0.0f;
		   
		   int T_max = T;
		   float W_ini = 0.3f;
		   float W_min = 0.1f;
		   float F_max = 10.0f;
		   
		      
		   // clear temp buffers
		   std::fill_n(temp1, N, 0.0f);
		   std::fill_n(temp2, N, 0.0f);
		   std::fill_n(temp3, N, 0.0f);		   
		   std::fill_n(temp4, N, 0.0f);		   
		   std::fill_n(temp5, N, 0.0f);	
		   std::fill_n(temp6, N, 0.0f);				   

		   // step 1: evaluate fitness function			   
		   vector<float> fitnessVal = EvaluateFitnessFunctionVector(N, Rk, Xp, Xt);
		   
		   // step 2: calculate adaptive weights using particle fitness func value
		   // 2.1 A=(1 + cos(t*Pi/t_max)) - result in temp2
		   memcpy(temp1, Xt, N*sizeof(float));
		   float step_value = (step*M_PI/T_max);
		   std::fill_n(temp1, N, step_value);				   
		   std::fill_n(temp2, N, 1.0f);
		   vector<float> vect2(temp1, temp1 + N);			   
		   std::transform(vect2.begin(), vect2.end(), vect2.begin(), (float (*)(float))cos);			   
		   alpha = 1.0f;			   
		   cblas_saxpy(N, alpha, &vect2[0], 1, temp2, 1);
		   
		   // 2.2 B=(1- f/f_max) - result in temp3
		   alpha = (-1.f/F_max);
		   std::fill_n(temp3, N, 1.0f);
		   cblas_saxpy(N, alpha, &fitnessVal[0], 1, temp3, 1);
		   
		   // 2.3 C=A*B*w_ini/2 + w_min
		   // setup diagonal matrix Diag(B)
		   std::fill_n(DXp, N*N, 0.0f);
		   for(int i=0; i<N; i++) {
			   DXp[Idx2C(i,i,N)] = temp3[i];
		   }	
		   // C = (w_ini/2)*(Diag(B))*A + w_min
		   alpha = (W_ini/2);
		   beta = 1.0f;
		   std::fill_n(temp6, N, W_min);
		   cblas_sgemv(CblasColMajor, CblasNoTrans, N, N, alpha, DXp, N, temp2, 1, beta, temp6, 1);	   					
		   // copy back result
		   memcpy(result, temp6, N*sizeof(float));			
			
	}
	catch(std::exception &e) {
		eptr = std::current_exception();
	}
	catch(...) {
		eptr = std::current_exception();
	}
	
	//cleanup
	if(Xt) free(Xt);
	if(Xp) free(Xp);
	if(DXp) free(DXp);
	
	if(temp1) free(temp1);
	if(temp2) free(temp2);
	if(temp3) free(temp3);
	if(temp4) free(temp4);
	if(temp5) free(temp5);
	if(temp6) free(temp6);
	
	//auto finish = std::chrono::high_resolution_clock::now();	
	//std::chrono::duration<double> elapsed = finish - start;
	//std::cout << "Calc Adaptive Weights:: " << elapsed.count() << "\n";
	   
	
	if(eptr)
	{
		std::rethrow_exception(eptr);
	}
	
	return result;
}

// check for memory leaks - DONE
float* AdaptiveSwarmOptimisedPFWorker::UpdateVelocity(int N, float* weightVector){
	str:exception_ptr eptr;
	float* Pt = 0;
	float* Xt = 0;
	float* Gt = 0;
	float* Vt = 0;
	float* Xp = 0;
	float* DXp = 0;
	float* DW = 0;	
	float* DX = 0;	
	float* DC1 = 0;	
	float* DC2 = 0;	
	float* DC3 = 0;	
	float* Rp = 0;
	float* Rt = 0;
	float* R1 = 0;
	float* R2 = 0;
	float* R3 = 0;
	float* Rs = 0;	
	float* result =0;
	
	float* temp1 = 0;
	float* temp2 = 0;
	float* temp3 = 0;	
	float* temp4 = 0;	
	float* temp5 = 0;
	float* temp6 = 0;
	float* temp7 = 0;
	float* temp8 = 0;
	float* temp9 = 0;
	float* temp10 = 0;
	
	int C1 = 2;
	int C2 = 2;
	float W = 5.0f;
	float Rk = 4.5f;
	
	std::vector<float> randomArray;
	
	//auto start = std::chrono::high_resolution_clock::now();		
	
	try {
		   Xt = (float*)malloc(N*sizeof(float));
		   Xp = (float*)malloc(N*sizeof(float));
		   DXp = (float*)malloc(N*N*sizeof(float));
		   DW = (float*)malloc(N*N*sizeof(float));
		   DX = (float*)malloc(N*N*sizeof(float));
		   DC1 = (float*)malloc(N*N*sizeof(float));
		   DC2 = (float*)malloc(N*N*sizeof(float));
		   DC3 = (float*)malloc(N*N*sizeof(float));
		   
		   Rp = (float*)malloc(sizeof(float));
		   Rt = (float*)malloc(N*sizeof(float));
		   Rs = (float*)malloc(sizeof(float));
		   Vt = (float*)malloc(N*sizeof(float));
		   Pt = (float*)malloc(N*sizeof(float));
		   Gt = (float*)malloc(N*sizeof(float));
		   
		   R1 = (float*)malloc(N*sizeof(float));
		   R2 = (float*)malloc(N*sizeof(float));
		   R3 = (float*)malloc(N*sizeof(float));
		   
		   temp1 = (float*)malloc(N*sizeof(float));
		   temp2 = (float*)malloc(N*sizeof(float));
		   temp3 = (float*)malloc(N*sizeof(float));
		   temp4 = (float*)malloc(N*sizeof(float));
		   temp5 = (float*)malloc(N*sizeof(float));
		   temp6 = (float*)malloc(N*sizeof(float));
		   temp7 = (float*)malloc(N*sizeof(float));
		   temp8 = (float*)malloc(N*sizeof(float));
		   temp9 = (float*)malloc(N*sizeof(float));
		   temp10 = (float*)malloc(N*sizeof(float));
		   result = (float*)malloc(N*sizeof(float));
		   
		   std::fill_n(Xt, N, 2.0f);
		   std::fill_n(Xp, N, 1.0f);
		   std::fill_n(Rp, 1, 3.0f);
		   std::fill_n(Rs, 1, 2.0f);
		   std::fill_n(Rt, N, 2.0f);
		   std::fill_n(Vt, N, 0.0f);
		   std::fill_n(Pt, N, 0.0f);
		   std::fill_n(Gt, N, 0.0f);
		   std::fill_n(result, N, 0.0f);
		   
		   float alpha = 1.0f;
		   float beta = 0.0f;
		   
		   float W_ini = 0.3f;
		   float W_min = 0.1f;
		   float F_max = 10.0f;		   
		     
		   // clear temp buffers
		   std::fill_n(temp1, N, 0.0f);
		   std::fill_n(temp2, N, 0.0f);
		   std::fill_n(temp3, N, 0.0f);		   
		   std::fill_n(temp4, N, 0.0f);		   
		   std::fill_n(temp5, N, 0.0f);	
		   std::fill_n(temp6, N, 0.0f);	
		   std::fill_n(temp7, N, 0.0f);	
		   std::fill_n(temp8, N, 0.0f);	
		   std::fill_n(temp9, N, 0.0f);	
		   std::fill_n(temp10, N, 0.0f);	
		   std::fill_n(DXp, N*N, 0.0f);
		   std::fill_n(DW, N*N, 0.0f);
		   std::fill_n(DX, N*N, 0.0f);
		   std::fill_n(DC1, N*N, 0.0f);
		   std::fill_n(DC2, N*N, 0.0f);
		   std::fill_n(DC3, N*N, 0.0f);			   
		   std::fill_n(R1, N, 0.0f);
		   std::fill_n(R2, N, 0.0f);
		   std::fill_n(R3, N, 0.0f);
	   
		   // step 1: calculate learning factors
		   // 1.1 fitness func values
		   vector<float> fitnessValVector = EvaluateFitnessFunctionVector(N, Rk, Xp, Xt);
		   
		   // 1.2 fit1
		   memcpy(temp1, &fitnessValVector[0], N*sizeof(float));
		   // 1.3 fit2
		   float maxFitnessVal = *max_element(fitnessValVector.begin(), fitnessValVector.end());
		   std::fill_n(temp2, N, maxFitnessVal);
		   // 1.4 fit3
		   vector<float> randomParticleVector = EvaluateFitnessFunctionVector(1, Rk, Rp, Rs);
		   std::fill_n(temp3, N, randomParticleVector[0]);			   
		   // 1.5 fit1 + fit2
		   memcpy(temp4, temp2, N*sizeof(float));
		   cblas_saxpy(N, alpha, temp1, 1, temp4, 1);
		   // 1.6 D = fit1 + fit2 + fit3
		   cblas_saxpy(N, alpha, temp3, 1, temp4, 1);			   
		   // 1.7 c1 = 2.8*fit1/(D)
		   alpha = 2.8;
		   cblas_sscal(N, alpha, temp1, 1);	
		   std::vector<float> a(temp1, temp1 + N);
		   std::vector<float> b(temp4, temp4 + N);			   
		   auto c1 = Divide( a, b ) ;
		   // 1.8 c2 = 2.8*fit2/(D)
		   cblas_sscal(N, alpha, temp2, 1);	
		   std::vector<float> g(temp2, temp2 + N);
		   auto c2 = Divide( g,  b ) ;
		   // 1.9 c3 = 2.8*fit3/(D)
		   cblas_sscal(N, alpha, temp3, 1);	
		   std::vector<float> h(temp3, temp3 + N);
		   auto c3 = Divide( h, b ) ;
		   
		   // step 2:: velocity update
		   // 2.1:: A = w*Vt  - get this using (Diag(W))*Vt			   
		   // setup diagonal matrix
		   for(int i=0; i<N; i++) {
			   DW[Idx2C(i,i,N)] = weightVector[i];
		   }	
		   
		   // calc as 1*(Diag(W))*Vt
		   alpha = 1.f;
		   cblas_sgemv(CblasColMajor, CblasNoTrans, N, N, alpha, DW, N, Vt, 1, beta, temp5, 1);
		   
		   // 2.2:: calc B = c1*r1*(Pbest - Xt)
		   // 2.2.1 X = c1*r1
		   randomArray.resize(N);
		   std::generate(randomArray.begin(), randomArray.end(), std::rand);		
		   memcpy(R1, &randomArray[0], N);
		   
		   // c1*r1  - get this using (Diag(c1))*r1			   
		   // setup diagonal matrix
		   for(int i=0; i<N; i++) {
			   DC1[Idx2C(i,i,N)] = c1[i];
		   }	
		   
		   // calc as 1*(Diag(c1))*r1
		   alpha = 1.f;
		   beta = 0.f;
		   cblas_sgemv(CblasColMajor, CblasNoTrans, N, N, alpha, DC1, N, R1, 1, beta, temp6, 1);
		   
		   // 2.2.2 Y = (Pbest - Xt)
		   alpha = -1.f;
		   memcpy(temp7, Pt, N*sizeof(float));
		   cblas_saxpy(N, alpha, Xt, 1, temp7, 1);
		   
		   // 2.2.3 X*Y - get this using (Diag(X))*Y			   
		   // setup diagonal matrix
		   for(int i=0; i<N; i++) {
			   DX[Idx2C(i,i,N)] = temp6[i];
		   }	
		   
		   // calc 1*(Diag(X))*Y
		   alpha = 1.f;
		   beta = 0.f;
		   cblas_sgemv(CblasColMajor, CblasNoTrans, N, N, alpha, DX, N, temp7, 1, beta, temp8, 1);
		   
		   // step 2.3:: C = c2*r2*(Gbest - Xt)
		   // 2.3.1 X = c2*r2
		   randomArray.resize(N);
		   std::generate(randomArray.begin(), randomArray.end(), std::rand);		
		   memcpy(R2, &randomArray[0], N);
			
		   // c2*r2  - get this using (Diag(c2))*r2			   
		   // setup diagonal matrix
		   for(int i=0; i<N; i++) {
			   DC2[Idx2C(i,i,N)] = c2[i];
		   }	
		   
		   // calc as 1*(Diag(c2))*r2
		   alpha = 1.f;
		   beta = 0.f;
		   std::fill_n(temp6, N, 0.0f);				   
		   cblas_sgemv(CblasColMajor, CblasNoTrans, N, N, alpha, DC2, N, R2, 1, beta, temp6, 1);
			
		   // 2.3.2 Y = (Gbest - Xt)
		   alpha = -1.f;
		   std::fill_n(temp7, N, 0.0f);			   
		   memcpy(temp7, Gt, N*sizeof(float));
		   cblas_saxpy(N, alpha, Xt, 1, temp7, 1);
		   
		   // 2.3.3 X*Y - get this using (Diag(X))*Y			   
		   // setup diagonal matrix
		   for(int i=0; i<N; i++) {
			   DX[Idx2C(i,i,N)] = temp6[i];
		   }	
		   
		   // calc 1*(Diag(X))*Y
		   alpha = 1.f;
		   beta = 0.f;
		   cblas_sgemv(CblasColMajor, CblasNoTrans, N, N, alpha, DX, N, temp7, 1, beta, temp9, 1);
		   
		   
		   // step 2.4:: D = c3*r3*(Rbest - Xt)
		   // 2.4.1 X = c3*r3
		   randomArray.resize(N);
		   std::generate(randomArray.begin(), randomArray.end(), std::rand);		
		   memcpy(R3, &randomArray[0], N);
			
		   // c3*r3  - get this using (Diag(c3))*r3			   
		   // setup diagonal matrix
		   for(int i=0; i<N; i++) {
			   DC3[Idx2C(i,i,N)] = c3[i];
		   }	
		   
		   // calc as 1*(Diag(c3))*r3
		   alpha = 1.f;
		   beta = 0.f;
		   std::fill_n(temp6, N, 0.0f);				   
		   cblas_sgemv(CblasColMajor, CblasNoTrans, N, N, alpha, DC3, N, R3, 1, beta, temp6, 1);
			
		   // 2.4.2 Y = (Rbest - Xt)
		   alpha = -1.f;
		   std::fill_n(temp7, N, 0.0f);			   
		   memcpy(temp7, Rt, N*sizeof(float));
		   cblas_saxpy(N, alpha, Xt, 1, temp7, 1);
		   
		   // 2.4.3 X*Y - get this using (Diag(X))*Y			   
		   // setup diagonal matrix
		   for(int i=0; i<N; i++) {
			   DX[Idx2C(i,i,N)] = temp6[i];
		   }	
		   
		   // calc 1*(Diag(X))*Y
		   alpha = 1.f;
		   beta = 0.f;
		   cblas_sgemv(CblasColMajor, CblasNoTrans, N, N, alpha, DX, N, temp7, 1, beta, temp10, 1);
		   
		   // step 2.5 K = A + B + C + D
		   // 2.5.1 p = A + B
		   cblas_saxpy(N, alpha, temp5, 1, temp8, 1);
		   
		   // 2.5.1 q = C + D
		   cblas_saxpy(N, alpha, temp9, 1, temp10, 1);
				
		   // 2.5.1 p + q
		   cblas_saxpy(N, alpha, temp8, 1, temp10, 1);
		   
		   // copy back result
		   memcpy(result, temp10, N*sizeof(float));	   
		   
		   // cleanup
		   fitnessValVector.clear();
		   randomParticleVector.clear();
		   a.clear();
		   b.clear();
		   g.clear();
		   h.clear();
		   c1.clear();
		   c2.clear();
		   c3.clear();
		   
	}
	catch(std::exception &e) {
		eptr = std::current_exception();
	}
	catch(...) {
		eptr = std::current_exception();
	}
	
	//cleanup
	if(Xt) free(Xt);
	if(Xp) free(Xp);	
	if(Pt) free(Pt);
	if(Gt) free(Gt);
	if(Vt) free(Vt);
	if(Rp) free(Rp);
	if(Rs) free(Rs);
	if(Rt) free(Rt);
	if(R1) free(R1);
	if(R2) free(R2);
	if(R3) free(R3);
	if(DXp) free(DXp);
	if(DX) free(DX);
	if(DW) free(DW);
	if(DC1) free(DC1);
	if(DC2) free(DC2);
	if(DC3) free(DC3);
	
	if(temp1) free(temp1);
	if(temp2) free(temp2);
	if(temp3) free(temp3);
	if(temp4) free(temp4);
	if(temp5) free(temp5);
	if(temp6) free(temp6);
	if(temp7) free(temp7);
	if(temp8) free(temp8);
	if(temp9) free(temp9);
	if(temp10) free(temp10);
	
	//auto finish = std::chrono::high_resolution_clock::now();	
	//std::chrono::duration<double> elapsed = finish - start;
	//std::cout << "Update Velocity:: " << elapsed.count() << "\n";
	
	
	if(eptr)
	{
		std::rethrow_exception(eptr);
	}
	
	return result;
}

// check for memory leaks - DONE
void AdaptiveSwarmOptimisedPFWorker::UpdatePosition(int N, float* velocityVector){
	str:exception_ptr eptr;
	
	float* Xt = 0;
	float* temp1 = 0;
	float alpha = 1.0f;
	
	//auto start = std::chrono::high_resolution_clock::now();	
	
	try {
		   Xt = (float*)malloc(N*sizeof(float));
		   temp1 = (float*)malloc(N*sizeof(float));
		   
		   std::fill_n(Xt, N, 2.0f);
		   std::fill_n(temp1, N, 0.0f);
		   
		   // calculate Xt + Vt : result stored in temp1
		   memcpy(temp1, velocityVector, N*sizeof(float));
		   cblas_saxpy(N, alpha, Xt, 1, temp1, 1);	
	}
	catch(std::exception &e) {
		eptr = std::current_exception();
	}
	catch(...) {
		eptr = std::current_exception();
	}
	
	//cleanup
	if(Xt) free(Xt);
	if(temp1) free(temp1);
	
	//auto finish = std::chrono::high_resolution_clock::now();	
	//std::chrono::duration<double> elapsed = finish - start;
	//std::cout << "Update Position:: " << elapsed.count() << "\n";
	
	if(eptr)
	{
		std::rethrow_exception(eptr);
	}	
	
};

// check for memory leaks - DONE
void AdaptiveSwarmOptimisedPFWorker::PFTimeUpdate(int p, int N, int randomNumComplexityConst){
	
	str:exception_ptr eptr;
	float* T1=0;
	float* T2=0;
	float* Ap=0;		
	float* Xp=0;
	float* W=0;
	float* M=0;
	float* temp2=0;
	float* temp3=0;
	float* temp4=0;
	float* temp5=0;	
	double* C=0;
	double *R=0;
	
	try {

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
		
		// allocate buffer space
		T1 = (float*)malloc(matrixT1Rows*matrixT1Cols*sizeof(float));
		T2 = (float*)malloc(matrixT2Rows*matrixT2Cols*sizeof(float));
		Ap = (float*)malloc(matrixApRows*matrixApCols*sizeof(float));
		Xp = (float*)malloc(matrixXpRows*matrixXpCols*sizeof(float));
		W = (float*)malloc(matrixWRows*matrixWCols*sizeof(float));
		M = (float*)malloc(matrixMRows*matrixMCols*sizeof(float));
		C = 0;					
		
		temp2 = (float*)malloc(matrixT2Rows*matrixT2Cols*sizeof(float));
		temp3 = (float*)malloc(matrixXpRows*matrixXpCols*sizeof(float));
		temp4 = (float*)malloc(matrixXpRows*matrixXpCols*sizeof(float));
		temp5 = (float*)malloc(matrixWRows*matrixWCols*sizeof(float));
		
		float alpha = 1.0f;
		float beta = 0.0f;
		
		// step 1 cholesky factorisation of M
		int num, info;
		char uplo = 'U';
		num = p; 		
		// allocate space 
		C = (double *)malloc(num*(num+1)*sizeof(double)/2);		
		R = GetSymmetricPostiveDefiniteMatrix(num);
		std::fill_n(C, num*(num+1)/2, 0.f);
		int i,j,h;
		h = 0;
		
		for (j = 0; j < num; j++){
			for (i = 0; i < num; i++){
				if(j >= i) {
					C[h] = R[Idx2C(i,j,num)];
					h +=1;
				}
			}
		}		
		dpptrf_(&uplo, &num, C, &info); 
	
		// step 2 - perform T_1*T_2 (resultant matrix is p x N)
		beta = 0.0f;
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, matrixT1Rows, matrixT2Cols, matrixT1Cols, alpha, T1, matrixT1Rows, T2, matrixT2Rows, beta, temp2, matrixT2Rows);					
						
		// step 3 - perform T3 = A_p*X_p (resultant matrix is p x N)
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, matrixApRows, matrixXpCols, matrixApCols, alpha, Ap, matrixApRows, Xp, matrixXpRows, beta, temp3, matrixXpRows);
									
		// step 4 - perform T3 + T4 + W
		// copy T3 to temp buffer
		alpha = 1.0f;
		memcpy(temp5, temp3, matrixXpRows*matrixXpCols*sizeof(float));
		cblas_saxpy(matrixXpRows*matrixXpCols, alpha, temp4, 1, temp5, 1);
		cblas_saxpy(matrixWRows*matrixWCols, alpha, W, 1, temp5, 1);
		
	}
	catch(std::exception &e) {
		eptr = std::current_exception();
	}
	catch(...) {
		eptr = std::current_exception();
	}	
		
	//cleanup
	if(T1) free(T1);
	if(C) free(C);
	if(R) free(R);
	if(T2) free(T2);
	if(Ap) free(Ap);
	if(Xp) free(Xp);
	if(W) free(W);
	if(M) free(M);
	
	if(temp2) free(temp2);
	if(temp3) free(temp3);
	if(temp4) free(temp4);
	if(temp5) free(temp5);		
	
	if(eptr)
	{
		std::rethrow_exception(eptr);
	}		
}

double* AdaptiveSwarmOptimisedPFWorker::GetSymmetricPostiveDefiniteMatrix(int n)
{
	double *L, *C, *B;
	L =  (double *)malloc(n*n*sizeof(double));
	C =  (double *)malloc(n*n*sizeof(double));
    
	std::fill_n(L, n*n, 0.f); 
	std::fill_n(C, n*n, 0.f);
	
	int i,j;
	for (j = 0; j < n; j++){
		for (i = 0; i < n; i++){
			if(j <= i) {
				L[Idx2C(i,j,n)]= 4;
			}
		}
	}
	
	float alpha = 1.f;
	float beta = 0.f;

	// compute matrix C = L<L^T>
	cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, n, n, alpha, L, n, beta, C, n); 
	
	free(L);
	return C;	
}

vector<float> AdaptiveSwarmOptimisedPFWorker::EvaluateFitnessFunctionVector(int N, float Rk, float* Y_new, float* Y_pred){
	str:exception_ptr eptr;
	float* DXp = 0;
	
	float* temp4 = 0;	
	float* temp5 = 0;
	vector<float> result;
	
	try {
		   DXp = (float*)malloc(N*N*sizeof(float));
		   temp4 = (float*)malloc(N*sizeof(float));
		   temp5 = (float*)malloc(N*sizeof(float));
		   std::fill_n(DXp, N*N, 0.0f);
		   
		   float alpha = -1.0f;
		   float beta = 0.0f;
		      
		   // clear temp buffers
		   std::fill_n(temp4, N, 0.0f);		   
		   std::fill_n(temp5, N, 0.0f);	
		   
		   // step 1: evaluate fitness function 
		   // step 3.1 Y_diff = Y_new - Y_pred
		   alpha = -1.0f;
		   memcpy(temp4, Y_new, N*sizeof(float));
		   cblas_saxpy(N, alpha, Y_pred, 1, temp4, 1);
		   
		   // step 3.2 (y_new - y_pred)^2 - get this using (Diag(Y_diff))*Y_diff			   
		   // setup diagonal matrix
		   for(int i=0; i<N; i++) {
			   DXp[Idx2C(i,i,N)] = temp4[i];
		   }	
		   
		   // calc B = (-0.5/Rk)*(Diag(Y_diff))*Y_diff
		   alpha = (-0.5f/Rk);
		   cblas_sgemv(CblasColMajor, CblasNoTrans, N, N, alpha, DXp, N, temp4, 1, beta, temp5, 1);
		   
		   // calc exp(b_i) 
		   vector<float> vect(temp5, temp5 + N);
		   // call the `float' version of exp func
		   std::transform(vect.begin(), vect.end(), vect.begin(), (float (*)(float))exp);

		   result = vect;		  
	}
	catch(std::exception &e) {
		eptr = std::current_exception();
	}
	catch(...) {
		eptr = std::current_exception();
	}
	
	//cleanup
	if(DXp) free(DXp);	
	if(temp4) free(temp4);
	if(temp5) free(temp5);
	
	if(eptr)
	{
		std::rethrow_exception(eptr);
	}
	
	return result;
}

std::vector<float> AdaptiveSwarmOptimisedPFWorker::Divide(std::vector<float>& a, std::vector<float>& b)
{
    std::vector<float> result ;

    std::size_t n = std::min( a.size(), b.size() ) ;
    std::transform( std::begin(a), std::begin(a)+n, std::begin(b),
                    std::back_inserter(result), std::divides<float>{} ) ;
    return result ;
};




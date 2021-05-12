#ifndef WORKERS_CPU_ADAPTIVESWARMOPTIMISEDPFWORKER_H
#define WORKERS_CPU_ADAPTIVESWARMOPTIMISEDPFWORKER_H

#include <memory>
#include <core/worker.h>
#include <workers/cpu/cpu_worker.h>
#include <workers/utility/utility_worker.h>
#include <cstdio>
#include <list>
#include <vector>


class AdaptiveSwarmOptimisedPFWorker: public Worker{
    public:
        AdaptiveSwarmOptimisedPFWorker(const BehaviourContainer& container, const std::string& inst_name);
        ~AdaptiveSwarmOptimisedPFWorker();

		void SetProperties (int standardPF_P, int numParticles_N, int numIterations_T, int randomNumComplexityConstant, int maxThreadsAllowable);
		void RunOnMeasurements ();
		void RunOnMeasurementsFlops ();
		const std::string &get_version() const { return version; }
		
	private:	
	    void RunUnitTests();
		float* CalculateAdaptiveWeights(int N, int T, int step);
		float* UpdateVelocity(int N, float* weightVector);
		void UpdatePosition(int N, float* velocityVector);
		std::vector<float> EvaluateFitnessFunctionVector(int N, float Rk, float* Y_new, float* Y_pred);
		void PFTimeUpdate(int p, int N, int randomNumComplexityConst);
		double* GetSymmetricPostiveDefiniteMatrix(int n);
		std::vector<float> Divide( std::vector<float>& a, std::vector<float>& b );
		const std::string version = "1.0.1";
		std::unique_ptr<Cpu_Worker> cpu_worker_; 
		int p; 
		int N;
		int T;
		int randomNumComplexityConst;
		int maxThreads;
};

#endif  


#include "cpustatworker.h"
#include <core/component.h>
#include <cmath>
#include <thread>

CpuStatWorker::CpuStatWorker(const BehaviourContainer& container, const std::string& inst_name) 
	: Worker(container, GET_FUNC, inst_name)
{
    cpu_worker_.reset(new Cpu_Worker(*this, "cpu_worker"));
}

CpuStatWorker::~CpuStatWorker(){
	cpu_worker_.reset();
}

void *PrintHello(void *threadid) {
	pthread_exit(NULL);
}
	
void CpuStatWorker::WorstCaseWorkload(
	double fixedOperations, 
	double operationsPerContact, 
	int contacts, 
	const std::string &fromName, 
	double cpuScalerGlobal, 
	bool applyToAllSockets
) {
	const int MILLION = 1000000;
	
	const double totalOperations = fixedOperations + operationsPerContact * contacts;
	const double targetOperations = cpuScalerGlobal*totalOperations;
	const double targetMillionsOfOperations = lround(targetOperations/MILLION);
	
	const int numberOfThreads  = applyToAllSockets ? std::thread::hardware_concurrency() : 1;
	cpu_worker_->MWIP(targetMillionsOfOperations, numberOfThreads, true);
};


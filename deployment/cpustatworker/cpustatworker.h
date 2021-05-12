#ifndef WORKERS_CPU_CPUSTATWORKER_H
#define WORKERS_CPU_CPUSTATWORKER_H

#include <memory>
#include <core/worker.h>
#include <workers/cpu/cpu_worker.h>
#include <workers/utility/utility_worker.h>


class CpuStatWorker : public Worker{
    public:
        CpuStatWorker(const BehaviourContainer& container, const std::string& inst_name);
        ~CpuStatWorker();
		const std::string &get_version() const { return version; }
	
        void WorstCaseWorkload(double fixedOperations, double operationsPerContact, int contacts, const std::string& fromName, double cpuScalerGlobal, bool applyToAllSockets);

    private:
        std::unique_ptr<Cpu_Worker> cpu_worker_;
		const std::string version = "1.0.0";
};

#endif  //WORKERS_CPU_CPUSTATWORKER_H

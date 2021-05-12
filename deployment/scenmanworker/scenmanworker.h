#ifndef WORKERS_SCENMANWORKER_H
#define WORKERS_SCENMANWORKER_H

#include <core/worker.h>

class ScenManWorker : public Worker {
public:

    ScenManWorker(const BehaviourContainer& container, const std::string& inst_name);
    ~ScenManWorker();
    const std::string &get_version() const { return version; }

	void PrintClockString(int day_minutes);
	void PrintEnvironmentState(std::string header_string, int day_minutes, bool is_daytime);

private:
    const std::string version = "0.0.0";

};

#endif  //WORKERS_SCENMANWORKER_H

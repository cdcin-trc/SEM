#include "scenmanworker.h"
#include <core/component.h>
#include<iostream> 
#include<string> // used for string and to_string() 


ScenManWorker::ScenManWorker(const BehaviourContainer& container, const std::string& inst_name) 
	: Worker(container, GET_FUNC, inst_name)
{
}

ScenManWorker::~ScenManWorker() {
}

void ScenManWorker::PrintClockString(int day_minutes) {
	auto work_id = get_new_work_id();
	
	std::string m_clkstring;
	int mins = day_minutes;
	int hours_floor = (int)(mins/60.0);
	std::string hours_clk = std::to_string(hours_floor);
	int min_past_hour = mins - (hours_floor*60);
	std::string minutes_clk = std::to_string(min_past_hour);
	
	if (hours_floor<10)
		m_clkstring = "0";
	else
		m_clkstring = "";

	m_clkstring += hours_clk+"h:";

	if (min_past_hour<10)
		m_clkstring += "0";

	m_clkstring += minutes_clk+"m:00s";
	
	const auto& message_str = get_arg_string_variadic("Environment state: clock= %s", m_clkstring.c_str());
	Log(GET_FUNC, Logger::WorkloadEvent::MESSAGE, work_id, message_str);

}

void ScenManWorker::PrintEnvironmentState(std::string header_string, int day_minutes, bool is_daytime) {
	auto work_id = get_new_work_id();
	
	std::string m_clkstring = header_string;
	int mins = day_minutes;
	int hours_floor = (int)(mins/60.0);
	std::string hours_clk = std::to_string(hours_floor);
	int min_past_hour = mins - (hours_floor*60);
	std::string minutes_clk = std::to_string(min_past_hour);
	
	if (hours_floor<10)
		m_clkstring += "0";

	m_clkstring += hours_clk+"h:";

	if (min_past_hour<10)
		m_clkstring += "0";

	m_clkstring += minutes_clk+"m:00s";
	
	const auto& message_str = get_arg_string_variadic("Environment state: clock=%s, is_daytime=%d", m_clkstring.c_str(), is_daytime);
	Log(GET_FUNC, Logger::WorkloadEvent::MESSAGE, work_id, message_str);

}
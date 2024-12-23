#pragma once

#include <dunes/core/launch_parameters.hpp>
#include <sthe/cu/stopwatch.hpp>
#include <cuda_runtime.h>

namespace dunes
{

void initializeTerrain(const LaunchParameters& t_launchParameters, const InitializationParameters& t_initializationParameters);
void addSandForCoverage(const LaunchParameters& t_launchParameters, int2 res, bool uniform, int radius, float amount);
void initializeWindWarping(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters);
void initializeVegetation(const LaunchParameters& t_launchParameters);
void rain(const LaunchParameters& t_launchParameters);

void moisture(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters);
void transport(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters);
void sediment(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters);
void getVegetationCount(LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters);
void vegetation(LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters, std::vector<sthe::cu::Stopwatch>& watches);
void venturi(const LaunchParameters& t_launchParameters);
void windWarping(const LaunchParameters& t_launchParameters);
void windShadow(const LaunchParameters& t_launchParameters);
void pressureProjection(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters);
void continuousSaltation(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters);
void continuousReptation(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters);
void avalanching(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters);
void bedrockAvalanching(const LaunchParameters& t_launchParameters);

float coverage(const LaunchParameters& t_launchParameters, unsigned int* coverageMap, int num_cells, float threshold);

}

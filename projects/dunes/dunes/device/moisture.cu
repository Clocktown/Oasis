#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include "common.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

	__global__ void evaporationKernel(Array2D<float4> terrainArray, Array2D<float> moistureArray) {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		float4 terrain{ terrainArray.read(cell) };
		float moisture{ moistureArray.read(cell) };

		const float sandMoistureRate = c_parameters.sandMoistureRate;
		const float soilMoistureRate = c_parameters.soilMoistureRate;
		const float iTerrainThicknessThreshold = c_parameters.iTerrainThicknessMoistureThreshold;
		const float sandFactor = 1.f - clamp(terrain.y * iTerrainThicknessThreshold, 0.f, 1.f);
		const float moistureRate = lerp(sandMoistureRate, soilMoistureRate, sandFactor);

		const float moistureEvaporationFactor = 1.f - clamp(terrain.w * 10.f, 0.f, 1.f);
		terrain.w = terrain.w * exp(- c_parameters.evaporationRate * c_parameters.deltaTime);
		moisture = moisture * exp(-c_parameters.moistureEvaporationScale * moistureRate * moistureEvaporationFactor * c_parameters.deltaTime);

		setBorderWaterLevelMin(cell, terrain, c_parameters.waterBorderLevel);
		setWaterLevelMin(cell, terrain, c_parameters.waterLevel);

		terrainArray.write(cell, terrain);
		moistureArray.write(cell, moisture);
	}

	__global__ void moistureKernel(Array2D<float4> terrainArray, Array2D<float> moistureArray) {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		float4 terrain{ terrainArray.read(cell) };
		float moisture{ moistureArray.read(cell) };
		const float sandMoistureRate = c_parameters.sandMoistureRate;
		const float soilMoistureRate = c_parameters.soilMoistureRate;
		const float iTerrainThicknessThreshold = c_parameters.iTerrainThicknessMoistureThreshold;
		const float terrainThickness = terrain.y + terrain.z;
		const float moistureCapacityConstant = c_parameters.moistureCapacityConstant;
		const float moistureCapacity = moistureCapacityConstant * clamp(terrainThickness * iTerrainThicknessThreshold, 0.f, 1.f);
		const float sandFactor = 1.f - clamp(terrain.y * iTerrainThicknessThreshold, 0.f, 1.f);
		const float moistureRate = moisture > (0.5f * moistureCapacity) ? 0.02f * lerp(sandMoistureRate, soilMoistureRate, sandFactor) : lerp(sandMoistureRate, soilMoistureRate, sandFactor);

		if (moisture > moistureCapacity) {
			terrain.w += moisture - moistureCapacity;
			moisture = moistureCapacity;
		}
		else {
			const float dMoisture = fminf(moistureRate * (moistureCapacity - moisture), terrain.w);
			terrain.w -= dMoisture;
			moisture += dMoisture;
		}

		setBorderWaterLevelMin(cell, terrain, c_parameters.waterBorderLevel);
		setWaterLevelMin(cell, terrain, c_parameters.waterLevel);

		terrainArray.write(cell, terrain);
		moistureArray.write(cell, moisture);
	}

	__global__ void initMoistureDiffusionKernel(Buffer<float> moistureBuffer) {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		moistureBuffer[getCellIndex(cell)] = 0.f;
	}

	__global__ void moistureDiffusionKernel(const Array2D<float> moistureArray, Buffer<float> moistureBuffer) {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const float prev = moistureArray.read(cell);
		const float diffusion_coefficient = fminf(0.5f * c_parameters.rGridScale * c_parameters.rGridScale * c_parameters.deltaTime, 0.25f);

		float next = 0.f;
		for(int i = 0; i < 4; ++i) {
			const int2 nextCell = getWrappedCell(cell + c_offsets[2 * i]);

			next += moistureArray.read(nextCell);
		}
		next = prev + diffusion_coefficient * (next - 4 * prev); 

		moistureBuffer[getCellIndex(cell)] = next;
	}

	__global__ void finishMoistureDiffusionKernel(Array2D<float> moistureArray, const Buffer<float> moistureBuffer) {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		moistureArray.write(cell, moistureBuffer[getCellIndex(cell)]);
	}

	void moisture(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters) {
		Buffer<float> diffusedMoistureBuffer{ t_launchParameters.tmpBuffer };

		evaporationKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.terrainMoistureArray);
		moistureKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.terrainMoistureArray);

		initMoistureDiffusionKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (diffusedMoistureBuffer);
		moistureDiffusionKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainMoistureArray, diffusedMoistureBuffer);
		finishMoistureDiffusionKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainMoistureArray, diffusedMoistureBuffer);
	}
}
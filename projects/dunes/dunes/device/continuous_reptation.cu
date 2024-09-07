#include "constants.cuh"
#include "kernels.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>
#include <sthe/config/debug.hpp>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <cstdio>

namespace dunes
{

__global__ void setupContinuousReptationKernel(Buffer<float> t_reptationBuffer)
{
	const int stride{ getGridStride1D() };

	for (int cellIndex{ getGlobalIndex1D() }; cellIndex < c_parameters.cellCount; cellIndex += stride)
	{
		t_reptationBuffer[cellIndex] = 0.0f;
	}
}

__global__ void continuousAngularReptationKernel(const Buffer<float> t_slabBuffer, Buffer<float> t_reptationBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	const float2 resistance{ c_parameters.resistanceArray.read(cell).x, c_parameters.resistanceArray.read(cell).y };
	const float windShadow{ resistance.x * c_parameters.reptationUseWindShadow };

	const float4 terrain{ c_parameters.terrainArray.read(cell) };
	const float terrainThickness = terrain.y + terrain.z;
	const float moistureCapacityConstant = c_parameters.moistureCapacityConstant;
	const float moistureCapacity = moistureCapacityConstant * clamp(terrainThickness * c_parameters.iTerrainThicknessMoistureThreshold, 0.f, 1.f);
	const float moisture{ 2 * clamp(c_parameters.moistureArray.read(cell) / (moistureCapacity + 1e-6f), 0.f, 1.f) - 1  };

	const float moistureFactor = moisture > 0.f ?
		1.5f - 1.35f * moisture :
		1.5f + 0.5f * moisture;

	const float moistureVegetationFactor = moisture > 0.f ?
		1.5f - 1.25f * moisture :
		1.5f + 0.5f * moisture;

	const float slab{ t_slabBuffer[cellIndex] };
	const float2 wind{ c_parameters.windArray.read(cell) };

	float baseAngle = c_parameters.avalancheAngle * exp(-slab * (1.f - windShadow) * length(wind) * c_parameters.reptationStrength);

	// Store precomputed angle
	t_reptationBuffer[cellIndex] = lerp(moistureFactor * baseAngle, moistureVegetationFactor * c_parameters.vegetationAngle, fmaxf(resistance.y, 0.f));
}

__global__ void noReptationKernel(Buffer<float> t_reptationBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	const float vegetation{ c_parameters.resistanceArray.read(cell).y };

	const float4 terrain{ c_parameters.terrainArray.read(cell) };
	const float terrainThickness = terrain.y + terrain.z;
	const float moistureCapacityConstant = c_parameters.moistureCapacityConstant;
	const float moistureCapacity = moistureCapacityConstant * clamp(terrainThickness * c_parameters.iTerrainThicknessMoistureThreshold, 0.f, 1.f);
	const float moisture{ 2 * clamp(c_parameters.moistureArray.read(cell) / (moistureCapacity + 1e-6f), 0.f, 1.f) - 1  };

	const float moistureFactor = moisture > 0.f ?
		1.5f - 1.35f * moisture :
		1.5f + 0.5f * moisture;

	const float moistureVegetationFactor = moisture > 0.f ?
		1.5f - 1.25f * moisture :
		1.5f + 0.5f * moisture;

	// Store precomputed angle
	t_reptationBuffer[cellIndex] = lerp(moistureFactor * c_parameters.avalancheAngle, moistureVegetationFactor * c_parameters.vegetationAngle, fmaxf(vegetation, 0.f));
}

__global__ void continuousReptationKernel(Buffer<float> t_slabBuffer, Buffer<float> t_reptationBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	const float4 terrain{ c_parameters.terrainArray.read(cell) };
	const float height{ terrain.x + terrain.y + terrain.z };

	const float slab{ t_slabBuffer[cellIndex] };
	const float wind{ length(c_parameters.windArray.read(cell)) };

	float change{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		const int2 nextCell{ getWrappedCell(cell + c_offsets[i]) };
		const float nextSlab{ t_slabBuffer[getCellIndex(nextCell)] };
		const float nextWind{ length(c_parameters.windArray.read(cell)) };

		const float4 nextTerrain{ c_parameters.terrainArray.read(nextCell) };
		const float nextHeight{ nextTerrain.x + nextTerrain.y + nextTerrain.z };

		const float heightDifference{ (nextHeight - height) * c_parameters.rGridScale * c_rDistances[i]};
		const float heightScale = abs(heightDifference);// fmaxf(c_parameters.avalancheAngle - abs(heightDifference), 0.f) / c_parameters.avalancheAngle;

		// Enforce symmetric additive and subtractive changes, avoiding any atomics
		float step = fmaxf(0.25f * heightScale * (slab + nextSlab) * (wind + nextWind) * c_parameters.reptationSmoothingStrength, 0.f);
        change += signbit(heightDifference) ? -fminf(step, terrain.y) : fminf(step, nextTerrain.y);
	}

	t_reptationBuffer[cellIndex] = change * 0.125;
}

__global__ void finishContinuousReptationKernel(Buffer<float> t_reptationBuffer)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
		{
			const int cellIndex{ getCellIndex(cell) };

			float4 terrain{ c_parameters.terrainArray.read(cell) };
			terrain.y += t_reptationBuffer[getCellIndex(cell)];

			c_parameters.terrainArray.write(cell, terrain);
		}
	}
}

void continuousReptation(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters)
{
	Buffer<float> reptationBuffer{ t_launchParameters.tmpBuffer + 4 * t_simulationParameters.cellCount };
	if (t_simulationParameters.reptationSmoothingStrength > 0.f) {
		continuousReptationKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.tmpBuffer, reptationBuffer);
		finishContinuousReptationKernel << <t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (reptationBuffer);
	}
	if (t_simulationParameters.reptationStrength > 0.f) {
		continuousAngularReptationKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.tmpBuffer, reptationBuffer);
	}
	else {
		noReptationKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (reptationBuffer);
	}
}

}

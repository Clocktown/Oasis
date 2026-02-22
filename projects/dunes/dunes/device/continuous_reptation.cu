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

__global__ void setupContinuousReptationKernel(Buffer<half> t_reptationBuffer)
{
	const int stride{ getGridStride1D() };

	for (int cellIndex{ getGlobalIndex1D() }; cellIndex < c_parameters.cellCount; cellIndex += stride)
	{
		t_reptationBuffer[cellIndex] = CUDART_ZERO_FP16;
	}
}

__global__ void continuousAngularReptationKernel(const Buffer<half> t_slabBuffer, Buffer<half> t_reptationBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	const float2 resistance{ __half22float2(c_parameters.resistanceArray.read(cell).a) };
	const float windShadow{ resistance.x * c_parameters.reptationUseWindShadow };

	const float4 terrain{ half4toFloat4(c_parameters.terrainArray.read(cell)) };
	const float terrainThickness = terrain.y + terrain.z;
	const float moistureCapacityConstant = c_parameters.moistureCapacityConstant;
	const float moistureCapacity = moistureCapacityConstant * clamp(terrainThickness * c_parameters.iTerrainThicknessMoistureThreshold, 0.f, 1.f);
	const float moisture{ 2.f * clamp(__half2float(c_parameters.moistureArray.read(cell)) / (moistureCapacity + 1e-6f), 0.f, 1.f) - 1.f  };

	const float moistureFactor = moisture > 0.f ?
		1.5f - 1.35f * moisture :
		1.5f + 0.5f * moisture;

	const float moistureVegetationFactor = moisture > 0.f ?
		1.5f - 1.25f * moisture :
		1.5f + 0.5f * moisture;

	const float slab{ t_slabBuffer[cellIndex] };
    const float2 wind {sampleLinearOrNearest<true>(c_parameters.windArray,
                                                    0.5f * (make_float2(cell) + 0.5f))};

	float baseAngle = c_parameters.avalancheAngle * exp(-slab * (1.f - windShadow) * length(wind) * c_parameters.reptationStrength);

	// Store precomputed angle
    t_reptationBuffer[cellIndex] =
            __float2half(c_parameters.gridScale *
                            lerp(moistureFactor * baseAngle,
                                moistureVegetationFactor * c_parameters.vegetationAngle,
                                fmaxf(resistance.y, 0.f)));
}

__global__ void noReptationKernel(Buffer<half> t_reptationBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	const float vegetation{ __half2float(c_parameters.resistanceArray.read(cell).a.y) };

	const float4 terrain{ half4toFloat4(c_parameters.terrainArray.read(cell)) };
	const float terrainThickness = terrain.y + terrain.z;
	const float moistureCapacityConstant = c_parameters.moistureCapacityConstant;
	const float moistureCapacity = moistureCapacityConstant * clamp(terrainThickness * c_parameters.iTerrainThicknessMoistureThreshold, 0.f, 1.f);
	const float moisture{ 2.f * clamp(__half2float(c_parameters.moistureArray.read(cell)) / (moistureCapacity + 1e-6f), 0.f, 1.f) - 1.f  };

	const float moistureFactor = moisture > 0.f ?
		1.5f - 1.35f * moisture :
		1.5f + 0.5f * moisture;

	const float moistureVegetationFactor = moisture > 0.f ?
		1.5f - 1.25f * moisture :
		1.5f + 0.5f * moisture;

	// Store precomputed angle
        t_reptationBuffer[cellIndex] =
                __float2half(c_parameters.gridScale *
                             lerp(moistureFactor * c_parameters.avalancheAngle,
                                  moistureVegetationFactor * c_parameters.vegetationAngle,
                                  fmaxf(vegetation, 0.f)));
}

__global__ void continuousReptationKernel(Buffer<half> t_slabBuffer, Buffer<half> t_reptationBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	const float4 terrain{ half4toFloat4(c_parameters.terrainArray.read(cell)) };
	const float height{ terrain.x + terrain.y + terrain.z };

	const float slab{ t_slabBuffer[cellIndex] };
    const float wind {length(
                sampleLinearOrNearest<true>(c_parameters.windArray, 0.5f * (make_float2(cell) + 0.5f)))};

	float change{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		const int2 nextCell{ getWrappedCell(cell + c_offsets[i]) };
		const float nextSlab{ t_slabBuffer[getCellIndex(nextCell)] };
                const float nextWind {length(sampleLinearOrNearest<true>(
                        c_parameters.windArray, 0.5f * (make_float2(nextCell) + 0.5f)))};

		const float4 nextTerrain{ half4toFloat4(c_parameters.terrainArray.read(nextCell)) };
		const float nextHeight{ nextTerrain.x + nextTerrain.y + nextTerrain.z };

		const float heightDifference{ (nextHeight - height) * c_parameters.rGridScale * c_rDistances[i]};
		const float heightScale = abs(heightDifference);// fmaxf(c_parameters.avalancheAngle - abs(heightDifference), 0.f) / c_parameters.avalancheAngle;

		// Enforce symmetric additive and subtractive changes, avoiding any atomics
		float step = fmaxf(0.25f * heightScale * (slab + nextSlab) * (wind + nextWind) * c_parameters.reptationSmoothingStrength, 0.f);
        change += signbit(heightDifference) ? -fminf(step, terrain.y) : fminf(step, nextTerrain.y);
	}

	t_reptationBuffer[cellIndex] = __float2half(change * 0.125f);
}

__global__ void finishContinuousReptationKernel(Buffer<half> t_reptationBuffer)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
		{
			const int cellIndex{ getCellIndex(cell) };

			half4 terrain{ c_parameters.terrainArray.read(cell) };
			terrain.a.y += t_reptationBuffer[getCellIndex(cell)];

			c_parameters.terrainArray.write(cell, terrain);
		}
	}
}

void continuousReptation(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters)
{
	Buffer<half> reptationBuffer{ t_launchParameters.tmpBuffer + 2 * t_simulationParameters.cellCount };
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

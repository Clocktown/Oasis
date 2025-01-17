#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void setupAtomicInPlaceAvalanchingKernel(Buffer<float2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	const float4 terrain{ c_parameters.terrainArray.read(cell) };

	t_terrainBuffer[cellIndex] = make_float2(terrain.x + terrain.z, terrain.y);
}

__global__ void atomicInPlaceAvalanchingKernel(Buffer<float2> t_terrainBuffer, const Buffer<float> t_reptationBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	const float avalancheAngle{ t_reptationBuffer[cellIndex] };

	const float2 terrain{ t_terrainBuffer[cellIndex] };
	const float height{ terrain.x + terrain.y };
	int nextCellIndices[8];
	float avalanches[8];
	float avalancheSum{ 0.0f };
	float maxAvalanche{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		nextCellIndices[i] = getCellIndex(getWrappedCell(cell + c_offsets[i]));

		const float2 nextTerrain{ t_terrainBuffer[nextCellIndices[i]] };
		const float nextHeight{ nextTerrain.x + nextTerrain.y };

		const float heightDifference{ height - nextHeight };
		avalanches[i] = fmaxf(heightDifference - avalancheAngle * c_distances[i] * c_parameters.gridScale, 0.0f);
		avalancheSum += avalanches[i];
		maxAvalanche = fmaxf(maxAvalanche, avalanches[i]);
	}

	if (avalancheSum > 0.0f)
	{
		const float rAvalancheSum{ 1.0f / avalancheSum };
		const float avalancheSize{ fminf(maxAvalanche / (1.0f + maxAvalanche * rAvalancheSum), terrain.y) };

		const float scale{ avalancheSize * rAvalancheSum };

		for (int i{ 0 }; i < 8; ++i)
		{
			if (avalanches[i] > 0.0f)
			{
				atomicAdd(&t_terrainBuffer[nextCellIndices[i]].y, scale * avalanches[i]);
			}
		}

		atomicAdd(&t_terrainBuffer[cellIndex].y, -avalancheSize);
	}
}

__global__ void finishAtomicInPlaceAvalanchingKernel(Buffer<float2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	float4 terrain = c_parameters.terrainArray.read(cell);
	terrain.y = t_terrainBuffer[cellIndex].y;

	c_parameters.terrainArray.write(cell, terrain);
}

void avalanching(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters)
{
	Buffer<float2> terrainBuffer{ (float2*)t_launchParameters.tmpBuffer };
	Buffer<float> reptationBuffer{ t_launchParameters.tmpBuffer + 4 * t_simulationParameters.cellCount };

	setupAtomicInPlaceAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(terrainBuffer);

	for (int i = 0; i < t_launchParameters.avalancheIterations; ++i)
	{
		atomicInPlaceAvalanchingKernel<< <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (terrainBuffer, reptationBuffer);
	}

	finishAtomicInPlaceAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(terrainBuffer);
}

}

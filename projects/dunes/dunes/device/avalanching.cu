#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void setupAtomicInPlaceAvalanchingKernel(Buffer<float> t_heightBuffer, Buffer<float> t_sandBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	const float4 terrain{ c_parameters.terrainArray.read(cell) };

	t_heightBuffer[cellIndex] = terrain.x + terrain.z;
	t_sandBuffer[cellIndex] = terrain.y;
}

__global__ void atomicInPlaceAvalanchingKernel(Buffer<float> t_heightBuffer, Buffer<float> t_sandBuffer, const Buffer<float> t_reptationBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	const float avalancheAngle{ t_reptationBuffer[cellIndex] };

	float sand{ t_sandBuffer[cellIndex] };
	float height{ t_heightBuffer[cellIndex] + sand };
	int nextCellIndices[8];
	float avalanches[8];
	float avalancheSum{ 0.0f };
	float maxAvalanche{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		nextCellIndices[i] = getCellIndex(getWrappedCell(cell + c_offsets[i]));
		const float nextHeight{ t_heightBuffer[nextCellIndices[i]] + t_sandBuffer[nextCellIndices[i]] };

		const float heightDifference{ height - nextHeight };
		avalanches[i] = fmaxf(heightDifference - avalancheAngle * c_distances[i] * c_parameters.gridScale, 0.0f);
		avalancheSum += avalanches[i];
		maxAvalanche = fmaxf(maxAvalanche, avalanches[i]);
	}

	if (avalancheSum > 0.0f)
	{
		const float rAvalancheSum{ 1.0f / avalancheSum };
		const float avalancheSize{ fminf(maxAvalanche / (1.0f + maxAvalanche * rAvalancheSum), sand) };


		const float scale{ avalancheSize * rAvalancheSum };

		for (int i{ 0 }; i < 8; ++i)
		{
			if (avalanches[i] > 0.0f)
			{
				atomicAdd(&t_sandBuffer[nextCellIndices[i]], scale * avalanches[i]);
			}
		}

		atomicAdd(&t_sandBuffer[cellIndex], -avalancheSize);
	}
}

__global__ void finishAtomicInPlaceAvalanchingKernel(Buffer<float> t_sandBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	float4 terrain = c_parameters.terrainArray.read(cell);
	terrain.y = t_sandBuffer[cellIndex];

	c_parameters.terrainArray.write(cell, terrain);
}

void avalanching(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters)
{
	Buffer<float> heightBuffer{ t_launchParameters.tmpBuffer };
	Buffer<float> sandBuffer{ t_launchParameters.tmpBuffer + t_simulationParameters.cellCount };
	Buffer<float> reptationBuffer{ t_launchParameters.tmpBuffer + 4 * t_simulationParameters.cellCount };

	setupAtomicInPlaceAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(heightBuffer, sandBuffer);

	for (int i = 0; i < t_launchParameters.avalancheIterations; ++i)
	{
		atomicInPlaceAvalanchingKernel<< <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (heightBuffer, sandBuffer, reptationBuffer);
	}

	finishAtomicInPlaceAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(sandBuffer);
}

}

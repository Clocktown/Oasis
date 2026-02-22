#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void setupAtomicInPlaceAvalanchingKernel(Buffer<half2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	const half4 terrain{ c_parameters.terrainArray.read(cell) };

	t_terrainBuffer[cellIndex] = half2(terrain.a.x + terrain.b.x, terrain.a.y);
}

__global__ void atomicInPlaceAvalanchingKernel(Buffer<half2> t_terrainBuffer, const Buffer<half> t_reptationBuffer)
{
    const int2 cell {getGlobalIndex2D()};
    if(isOutside(cell))
    {
        return;
    }
    const int   cellIndex {getCellIndex(cell)};
    const float avalancheAngle {__half2float(__ldg(&t_reptationBuffer[cellIndex]))};
    const half2 terrain {t_terrainBuffer[cellIndex]};
    const float height {__half2float(terrain.x + terrain.y)};
    float       avalanches[8];
    float       avalancheSum {0.0f};
    float       maxAvalanche {0.0f};
    #pragma unroll
    for(int i {0}; i < 8; ++i)
    {
        int         nextCell = getCellIndex(getWrappedCell(cell + c_offsets[i]));
        const half2 nextTerrain {t_terrainBuffer[nextCell]};
        const float nextHeight {__half2float(nextTerrain.x + nextTerrain.y)};
        const float heightDifference {height - nextHeight};
        avalanches[i] = fmaxf(heightDifference - avalancheAngle * c_distances[i], 0.0f);
        avalancheSum += avalanches[i];
        maxAvalanche = avalanches[i] > maxAvalanche ? avalanches[i] : maxAvalanche;
    }
    if(avalancheSum > 0.0f)
    {
        float avalancheSize;
        avalancheSize = fminf((maxAvalanche * avalancheSum) / (maxAvalanche + avalancheSum), __half2float(terrain.y));
        const float scale {avalancheSize / avalancheSum};
        #pragma unroll
        for(int i {0}; i < 8; ++i)
        {
            if(avalanches[i] > 0.0f)
            {
                atomicAdd(&t_terrainBuffer[getCellIndex(getWrappedCell(cell + c_offsets[i]))].y,
                          __float2half(scale * avalanches[i]));
            }
        }
        atomicAdd(&t_terrainBuffer[cellIndex].y, __float2half(-avalancheSize));
    }
}

__global__ void finishAtomicInPlaceAvalanchingKernel(Buffer<half2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	half4 terrain = c_parameters.terrainArray.read(cell);
	terrain.a.y = t_terrainBuffer[cellIndex].y;

	c_parameters.terrainArray.write(cell, terrain);
}

void avalanching(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters)
{
    Buffer<half2> terrainBuffer {reinterpret_cast<Buffer<half2>>(t_launchParameters.tmpBuffer)};
    Buffer<half>  reptationBuffer {t_launchParameters.tmpBuffer +
                                  2 * t_simulationParameters.cellCount};

	setupAtomicInPlaceAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(terrainBuffer);

	for (int i = 0; i < t_launchParameters.avalancheIterations; ++i)
	{
		atomicInPlaceAvalanchingKernel<< <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (terrainBuffer, reptationBuffer);
	}

	finishAtomicInPlaceAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(terrainBuffer);
}

}

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

#define TILE_SIZE 64
#define SHARED_SIZE (TILE_SIZE + 2)

__device__ __forceinline__ void cache_internal(float2 dst[SHARED_SIZE][SHARED_SIZE], Buffer<float2> src, int2 bid, int2 tid) {
	dst[tid.y + 1][tid.x + 1] = src[getCellIndex(bid * TILE_SIZE + tid)];
}

__device__ __forceinline__ void cache_border(float2 dst[SHARED_SIZE][SHARED_SIZE], Buffer<float2> src, int2 bid, int2 tid) {
	dst[tid.y + 1][tid.x + 1] = src[getCellIndex(getWrappedCell(bid * TILE_SIZE + tid))];
}

__device__ __forceinline__ float2& fetch_from_cache(float2 src[SHARED_SIZE][SHARED_SIZE], int2 tid) {
	return src[tid.y + 1][tid.x + 1];
}

__global__ void sharedAvalanchingKernel(Buffer<float2> t_terrainBuffer, const Buffer<float> t_reptationBuffer) {
	__shared__ float2 s_terrain[SHARED_SIZE][SHARED_SIZE];

	int2 bid = getBlockIndex2D();
	int2 tid = getThreadIndex2D();
	int lid = tid.x + tid.y * 16;
	
	for (int i = 0; i < TILE_SIZE; i += 16) {
		for (int j = 0; j < TILE_SIZE; j += 16) {
			cache_internal(s_terrain, t_terrainBuffer, bid, tid + make_int2(j, i));
		}
	}

	if (tid.y < 2) {
		cache_border(s_terrain, t_terrainBuffer, bid, make_int2(lid, -1));

		if (tid.x < 2) {
			cache_border(s_terrain, t_terrainBuffer, bid, (TILE_SIZE + 1) * make_int2(tid.y, tid.x) - 1);
		}
	}
	else if (tid.y < 4) {
		cache_border(s_terrain, t_terrainBuffer, bid, make_int2(lid & 31, TILE_SIZE));
	}
	else if (tid.y < 6) {
		cache_border(s_terrain, t_terrainBuffer, bid, make_int2(-1, lid & 31));
	}
	else if (tid.y < 8) {
		cache_border(s_terrain, t_terrainBuffer, bid, make_int2(TILE_SIZE, lid & 31));
	}
	else if (tid.y < 10) {
		cache_border(s_terrain, t_terrainBuffer, bid, make_int2(32 + (lid & 31), -1));
	}
	else if (tid.y < 12) {
		cache_border(s_terrain, t_terrainBuffer, bid, make_int2(32 + (lid & 31), TILE_SIZE));
	}
	else if (tid.y < 14) {
		cache_border(s_terrain, t_terrainBuffer, bid, make_int2(-1, 32 + (lid & 31)));
	}
	else {
		cache_border(s_terrain, t_terrainBuffer, bid, make_int2(TILE_SIZE, 32 + (lid & 31)));
	}

	__syncthreads();

	for (int i = 0; i < TILE_SIZE; i += 16) {
		for (int j = 0; j < TILE_SIZE; j += 16) {
			int2 id = tid + make_int2(j, i);
			float2 terrain = fetch_from_cache(s_terrain, id);
			float height = terrain.x + terrain.y;
			float angle = t_reptationBuffer[getCellIndex(bid * TILE_SIZE + id)];

			float avalanches[8];
			float avalancheSum = 0.0f;
			float maxAvalanche = 0.0f;

			for (int k = 0; k < 8; ++k) {
				int2 neighbor = id + c_offsets[k];
				float2 neighborTerrain = fetch_from_cache(s_terrain, neighbor);
				float neighborHeight = neighborTerrain.x + neighborTerrain.y;
				float delta = height - neighborHeight;

				avalanches[k] = fmaxf(delta - angle * c_distances[k] * c_parameters.gridScale, 0.0f);
				avalancheSum += avalanches[k];
				maxAvalanche = fmaxf(maxAvalanche, avalanches[k]);
			}

			if (avalancheSum > 0.0f)
			{
				float rAvalancheSum = 1.0f / avalancheSum;
				float avalancheSize = fminf(maxAvalanche / (1.0f + maxAvalanche * rAvalancheSum), terrain.y);

				float scale = avalancheSize * rAvalancheSum;

				for (int k = 0; k < 8; ++k) {
					if (avalanches[k] > 0.0f) {
						atomicAdd(&t_terrainBuffer[getCellIndex(getWrappedCell(bid * TILE_SIZE + id + c_offsets[k]))].y, scale * avalanches[k]);
					}
				}

				atomicAdd(&t_terrainBuffer[getCellIndex(bid * TILE_SIZE + id)].y, -avalancheSize);
			}
		}
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

	//uint3 gridDim;
	//gridDim.x = t_simulationParameters.gridSize.x / TILE_SIZE;
	//gridDim.y = t_simulationParameters.gridSize.y / TILE_SIZE;
	//gridDim.z = 1;

	//uint3 blockDim;
	//blockDim.x = 16;
	//blockDim.y = 16;
	//blockDim.z = 1;

	//for (int i = 0; i < t_launchParameters.avalancheIterations; ++i) {
	//	sharedAvalanchingKernel<<<gridDim, blockDim>>>(terrainBuffer, reptationBuffer);
	//}

	finishAtomicInPlaceAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(terrainBuffer);
}

}

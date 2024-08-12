#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>
#include <glm/glm.hpp>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <sthe/config/debug.hpp>

#include "random.cuh"

namespace dunes {

	__forceinline__ __device__ float getVegetationDensity(const Vegetation& veg, const float3& pos) {
		const float r2 = 0.25 * veg.radius * veg.radius;
		const float stem2 = 0.25 * veg.height.x * veg.height.x;
		const float root2 = 0.25 * veg.height.y * veg.height.y;
		const float3 covarStem{ 1.f / r2, 1.f / r2, 1.f / stem2 };
		const float3 covarRoot{ covarStem.x, covarStem.y, 1.f / root2 };
		const float3 covar = pos.z >= veg.pos.z ? covarStem : covarRoot;

		// Compute distance vector while considering toroidal boundaries
		float3 d = abs(pos - veg.pos);
		const float2 dims{ make_float2(c_parameters.gridSize) * c_parameters.gridScale };
		d.x = fminf(d.x, fabs(d.x - dims.x));
		d.y = fminf(d.y, fabs(d.y - dims.y));

		const float scale = 1.f; // peak vegetation density
		// gaussian distribution faded toward 0 at veg.radius
		return  fmaxf(scale * expf(-0.5f * dot(d, covar * d)) -  (length(float2{pos.x - veg.pos.x, pos.y - veg.pos.y}) / veg.radius) * expf(-2.f), 0.f);

	}

	__global__ void rasterizeVegetation(const Array2D<float4> t_terrainArray, Array2D<float4> t_resistanceArray, Buffer<Vegetation> vegBuffer, const Buffer<uint2> uniformGrid, Buffer<int> vegCount, int maxVegCount, Buffer<uint4> seeds)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		float4 resistance = t_resistanceArray.read(cell);
		if (resistance.y < 0.f) {
			return;
		}
		resistance.y = 0.f;


		const float2 position{ (make_float2(cell) + 0.5f) * c_parameters.gridScale };
		const float2 gridPosition{ position * c_parameters.rUniformGridScale };
		const int xStart = int(gridPosition.x - 0.5f);
		const int xEnd = int(gridPosition.x + 0.5f);
		const int yStart = int(gridPosition.y - 0.5f);
		const int yEnd = int(gridPosition.y + 0.5f);
		const float4 terrain = t_terrainArray.read(cell);
		const float3 pos{ position.x, position.y, terrain.x + terrain.y + terrain.z };

		for (int i = xStart; i <= xEnd; ++i) {
			for (int j = yStart; j <= yEnd; ++j) {
				const uint2 indices = uniformGrid[getCellIndex(getWrappedCell(int2{ i,j }, c_parameters.uniformGridSize), c_parameters.uniformGridSize)];
				for (unsigned int k = indices.x; k < indices.y && resistance.y < 1.f; ++k) {
					resistance.y += getVegetationDensity(vegBuffer[k], pos);
				}
			}
		}


		// TODO: Super simplistic algorithm.
		if (resistance.y == 0.f && *vegCount < maxVegCount) {
			int idx = getCellIndex(cell);
			unsigned int seed = seeds[idx].x;
			random::pcg(seed);
			seeds[idx].x = seed;
			const int xi = (seed % 10000000);
			if (xi < 100) {
				Vegetation veg;
				veg.type = 0;
				veg.pos = pos;
				veg.height = { 10.f, 10.f };
				veg.radius = 1.f;
				int oldIndex = atomicAdd(vegCount, 1);
				if (oldIndex < maxVegCount - 1) {
					vegBuffer[oldIndex] = veg;
				}
			}
		}

		resistance.y = fminf(resistance.y, 1.f);
		t_resistanceArray.write(cell, resistance);
	}

	__global__ void growVegetation(Buffer<Vegetation> vegBuffer, int vegCount, const Array2D<float4> t_terrainArray, const Buffer<uint2> uniformGrid)
	{
		const int idx = getGlobalIndex1D();
		if (idx >= vegCount) {
			return;
		}

		Vegetation veg = vegBuffer[idx];
		const float2 gridPos = { veg.pos.x * c_parameters.rGridScale, veg.pos.y * c_parameters.rGridScale };
		const float2 uniformGridPos = { veg.pos.x * c_parameters.rUniformGridScale, veg.pos.y * c_parameters.rUniformGridScale };
		const int xStart = int(uniformGridPos.x - 0.5f);
		const int xEnd = int(uniformGridPos.x + 0.5f);
		const int yStart = int(uniformGridPos.y - 0.5f);
		const int yEnd = int(uniformGridPos.y + 0.5f);
		const int2 cell = make_int2(gridPos); // for read from terrainArray if necessary

		float overlap = 0.f;

		for (int i = xStart; i <= xEnd; ++i) {
			for (int j = yStart; j <= yEnd; ++j) {
				const uint2 indices = uniformGrid[getCellIndex(getWrappedCell(int2{ i,j }, c_parameters.uniformGridSize), c_parameters.uniformGridSize)];
				for (unsigned int k = indices.x; k < indices.y; ++k) {
					if (k == idx) continue;
					// TODO: Very adhoc formula. Doesn't really consider the volume. Read literature what they actually do.
					float d = -0.5f * fminf(length(vegBuffer[k].pos - veg.pos) - (veg.radius + vegBuffer[k].radius), 0); // TODO: pos is not centered right now. Maybe have it centered and change height? or compute center here?
					d /= veg.radius;
					overlap += d * d;
				}
			}
		}

		// TODO: grow height? Remove height entirely?
		vegBuffer[idx].radius = fminf(veg.radius + fmaxf(1.f - overlap, 0.f) * c_parameters.deltaTime * 0.1f, 20.f); // TODO: 20.f = max radius
	}


	// temporary random fill
	__global__ void initVegetation(Buffer<Vegetation> vegBuffer, Buffer<uint4> seeds, int vegCount, Array2D<float4> t_terrainArray) {
		const int idx = getGlobalIndex1D();
		if (idx >= vegCount) {
			return;
		}

		uint4 seed = seeds[idx];
		Vegetation veg;
		veg.type = 0;
		random::pcg(seed);
		seeds[idx] = seed;
		const int2 vegCell{ seed.x % c_parameters.gridSize.x, seed.y % c_parameters.gridSize.y };
		const float4 terrain = t_terrainArray.read(vegCell);
		veg.pos = { (vegCell.x + 0.5f) * c_parameters.gridScale, (vegCell.y + 0.5f) * c_parameters.gridScale, terrain.x + terrain.y + terrain.z };
		const float height = 0.5f + (seed.z % 100000u) * 0.0003f;
		veg.height = { 0.6666f * height, 0.3333f * height };
		veg.radius = 1.f + (seed.w % 100000u) * 0.00019f;
		vegBuffer[idx] = veg;
	}

	__global__ void initUniformGrid(Buffer<uint2> uniformGrid) {
		const int idx = getGlobalIndex1D();
		if (idx >= c_parameters.uniformGridCount) {
			return;
		}

		uniformGrid[idx] = { 0,0 };
	}

	__global__ void fillKeys(const Buffer<Vegetation> vegBuffer, Buffer<unsigned int> keys, int count) {
		const int idx = getGlobalIndex1D();
		if (idx >= count) {
			return;
		}

		const auto pos = vegBuffer[idx].pos;
		const int2 uniformCell = make_int2(float2{ pos.x * c_parameters.rUniformGridScale, pos.y * c_parameters.rUniformGridScale });
		const int uniformIdx = getCellIndex(getWrappedCell(uniformCell, c_parameters.uniformGridSize), c_parameters.uniformGridSize);
		keys[idx] = uniformIdx;
	}

	__global__ void findGridStart(const Buffer<unsigned int> keys, Buffer<uint2> uniformGrid, int count) {
		const int idx = getGlobalIndex1D();
		if (idx >= count) {
			return;
		}

		const bool isZero = idx == 0;
		const unsigned int uniformIdxA = keys[idx];
		const unsigned int uniformIdxB = isZero ? 0 : keys[idx - 1];

		if (isZero || uniformIdxB != uniformIdxA) {
			uniformGrid[uniformIdxA].x = idx;
		}
	}

	__global__ void findGridEnd(const Buffer<unsigned int> keys, Buffer<uint2> uniformGrid, int count) {
		const int idx = getGlobalIndex1D();
		if (idx >= count) {
			return;
		}

		const bool isLast = idx == (count - 1);
		const unsigned int uniformIdxA = keys[idx];
		const unsigned int uniformIdxB = isLast ? 0 : keys[idx + 1];

		if (isLast || uniformIdxB != uniformIdxA) {
			uniformGrid[uniformIdxA].y = idx + 1;
		}
	}

	void getVegetationCount(LaunchParameters& t_launchParameters) {
		int count = 0;
		cudaMemcpy(&count, t_launchParameters.vegetationCount, sizeof(int), cudaMemcpyDeviceToHost);
		count = min(count, t_launchParameters.maxVegetation);
		std::cout << count << std::endl;
		t_launchParameters.numVegetation = count;
		t_launchParameters.vegetationGridSize1D = static_cast<unsigned int>(glm::ceil(static_cast<float>(count) / static_cast<float>(t_launchParameters.blockSize1D)));
	}

	void initializeVegetation(const LaunchParameters& t_launchParameters) {
		int count = t_launchParameters.numVegetation;
		initVegetation << < t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (t_launchParameters.vegBuffer, t_launchParameters.seedBuffer, count, t_launchParameters.terrainArray);
	}

	void vegetation(const LaunchParameters& t_launchParameters) {
		int count = t_launchParameters.numVegetation;
		initUniformGrid << <t_launchParameters.uniformGridSize1D, t_launchParameters.blockSize1D >> > (t_launchParameters.uniformGrid);
		fillKeys << <t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (t_launchParameters.vegBuffer, t_launchParameters.keyBuffer, count);
		thrust::sort_by_key(thrust::device, t_launchParameters.keyBuffer, t_launchParameters.keyBuffer + count, t_launchParameters.vegBuffer);
		findGridStart << <t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (t_launchParameters.keyBuffer, t_launchParameters.uniformGrid, count);
		findGridEnd << <t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (t_launchParameters.keyBuffer, t_launchParameters.uniformGrid, count);

		growVegetation << <t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (t_launchParameters.vegBuffer, count, t_launchParameters.terrainArray, t_launchParameters.uniformGrid);
		rasterizeVegetation << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.resistanceArray, t_launchParameters.vegBuffer, t_launchParameters.uniformGrid, t_launchParameters.vegetationCount, t_launchParameters.maxVegetation, t_launchParameters.seedBuffer);
	}
}
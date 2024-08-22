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
		const float2 height = c_vegTypes[veg.type].height;
		const float stem2 = height.x * height.x * r2;
		const float root2 = height.y * height.y * r2;
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
					const float density = getVegetationDensity(vegBuffer[k], pos);
					const bool isAlive = vegBuffer[k].health > 0.f;
					resistance.y += isAlive ? density : 0.f;
					resistance.w += isAlive ? 0.f : density;
				}
			}
		}


		// TODO: Super simplistic algorithm.
		if (resistance.y == 0.f && *vegCount < maxVegCount) {
			int idx = getCellIndex(cell);
			uint2 seed{ seeds[idx].x, seeds[idx].y };
			random::pcg(seed);
			seeds[idx].x = seed.x;
			seeds[idx].y = seed.y;
			const float xi = random::uniform_float(seed.x);
			if (xi < 0.00001f) {
				Vegetation veg;
				veg.type = seed.y % 2;
				veg.pos = pos;
				veg.age = 0.f;
				veg.health = 1.f;
				const float maxRadius = fminf(fmaxf(veg.pos.z - terrain.x, 0.f) / c_vegTypes[veg.type].height.y, c_vegTypes[veg.type].maxRadius);

				veg.radius = 0.05f * maxRadius;
				int oldIndex = atomicAdd(vegCount, 1);
				if (oldIndex < maxVegCount - 1) {
					vegBuffer[oldIndex] = veg;
				}
			}
		}

		resistance.y = fminf(resistance.y, 1.f);
		t_resistanceArray.write(cell, resistance);
	}

	__global__ void growVegetation(Buffer<Vegetation> vegBuffer, int vegCount, const Array2D<float4> t_terrainArray, const Buffer<uint2> uniformGrid, const Buffer<float> slopeBuffer, const Array2D<float> moistureArray)
	{
		const int idx = getGlobalIndex1D();
		if (idx >= vegCount) {
			return;
		}

		if (vegBuffer[idx].health <= 0.f || vegBuffer[idx].radius <= 1e-6f) {
			vegBuffer[idx].health = -1.f;
		}
		else {
			Vegetation veg = vegBuffer[idx];

			const float2 gridPos = { veg.pos.x * c_parameters.rGridScale, veg.pos.y * c_parameters.rGridScale };
			const float2 uniformGridPos = { veg.pos.x * c_parameters.rUniformGridScale, veg.pos.y * c_parameters.rUniformGridScale };
			const int xStart = int(uniformGridPos.x - 0.5f);
			const int xEnd = int(uniformGridPos.x + 0.5f);
			const int yStart = int(uniformGridPos.y - 0.5f);
			const int yEnd = int(uniformGridPos.y + 0.5f);
			const int2 cell = make_int2(gridPos); // for read from terrainArray if necessary
			const float4 terrain = t_terrainArray.read(cell);
			const float moisture = moistureArray.read(cell);
			const float slope = 2 * slopeBuffer[getCellIndex(cell)] - 1;
			const float bedrockHeight = terrain.x;
			const float soilHeight = bedrockHeight + terrain.z;
			const float sandHeight = soilHeight + terrain.y;
			const float waterLevel = sandHeight + terrain.w;

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
			// Age plant
			veg.age += c_parameters.deltaTime;

			// Plant parameters
			const float plantHeight = (veg.radius * c_vegTypes[veg.type].height.x);
			const float plantDepth = (veg.radius * c_vegTypes[veg.type].height.y);
			const bool isWaterPlant = c_vegTypes[veg.type].waterResistance >= 1.f;

			// Soil/Sand root conditions
			const float plantBottom = veg.pos.z - plantDepth;
			const float plantTop = veg.pos.z + plantHeight;
			const float soilCoverage = clamp((fminf(soilHeight, veg.pos.z) - fmaxf(plantBottom, bedrockHeight)) / plantDepth, 0.f, 1.f);
			const float sandCoverage = clamp((fminf(sandHeight, veg.pos.z) - fmaxf(plantBottom, soilHeight)) / plantDepth, 0.f, 1.f);
			const float soilRate = soilCoverage * c_vegTypes[veg.type].soilCompatibility + sandCoverage * c_vegTypes[veg.type].sandCompatibility;
			const float rootCoverage = clamp((fminf(sandHeight, veg.pos.z) - fmaxf(plantBottom, bedrockHeight)) / plantDepth, 0.f, 1.f);
			const float stemCoverage = clamp((fminf(sandHeight, plantTop) - fmaxf(veg.pos.z, bedrockHeight)) / plantHeight, 0.f, 1.f);
			const float rootDamage = -(rootCoverage - c_vegTypes[veg.type].terrainCoverageResistance.x) / c_vegTypes[veg.type].terrainCoverageResistance.x;
			const float stemDamage = (stemCoverage - c_vegTypes[veg.type].terrainCoverageResistance.y) / (1 - c_vegTypes[veg.type].terrainCoverageResistance.y);

			// Surface water conditions
			const float waterOverlap = clamp((waterLevel - fmaxf(veg.pos.z, sandHeight)) / plantHeight, 0.f, 1.f);
			const float waterRate = (waterOverlap - c_vegTypes[veg.type].waterResistance);
			const float waterDamage = isWaterPlant ? 0.f : waterRate / (1.f - c_vegTypes[veg.type].waterResistance);
			const float waterGrowth = isWaterPlant ? 1.f : fmaxf(-waterRate / c_vegTypes[veg.type].waterResistance, 0.f);


			// Calculate growth and health
			const float growthRate = soilRate * waterGrowth * fmaxf(1.f - overlap, 0.f) * c_parameters.deltaTime * c_vegTypes[veg.type].growthRate;
			veg.health -= 0.1f * c_parameters.deltaTime * (fmaxf(waterDamage, 0.f) + fmaxf(rootDamage, 0.f) + fmaxf(stemDamage, 0.f));
			veg.health += growthRate;

			// Calculate maximum radius based on root depth and bedrock and grow plant
			const float maxRadius = fminf(fmaxf(veg.pos.z - bedrockHeight, 0.f) / c_vegTypes[veg.type].height.y, c_vegTypes[veg.type].maxRadius);
			const float newRadius = fminf(veg.radius + growthRate, maxRadius);
			veg.radius = newRadius > veg.radius ? newRadius : veg.radius;

			if (isWaterPlant) {
				// water plant, force it to not grow outside of water
				veg.radius = fminf(veg.radius, fmaxf((waterLevel - veg.pos.z) / c_vegTypes[veg.type].height.x, 0.f));
			}
			// Check if maturity condition met
			if (veg.age > c_vegTypes[veg.type].maxMaturityTime && veg.radius < c_vegTypes[veg.type].maturityPercentage * c_vegTypes[veg.type].maxRadius) {
				veg.health = 0.f;
			}
			veg.health = clamp(veg.health, 0.f, 1.f);
			vegBuffer[idx] = veg;
		}
	}


	// temporary random fill
	__global__ void initVegetation(Buffer<Vegetation> vegBuffer, Buffer<uint4> seeds, int vegCount, Array2D<float4> t_terrainArray) {
		const int idx = getGlobalIndex1D();
		if (idx >= vegCount) {
			return;
		}

		uint4 seed = seeds[idx];
		random::pcg(seed);
		seeds[idx] = seed;

		Vegetation veg;
		veg.type = seed.w % 2;
		veg.age = 0.f;
		veg.health = 1.f;
		const int2 vegCell{ seed.x % c_parameters.gridSize.x, seed.y % c_parameters.gridSize.y };
		const float4 terrain = t_terrainArray.read(vegCell);
		veg.pos = { (vegCell.x + 0.5f) * c_parameters.gridScale, (vegCell.y + 0.5f) * c_parameters.gridScale, terrain.x + terrain.y + terrain.z };
		const float maxRadius = fminf(fmaxf(veg.pos.z - terrain.x, 0.f) / c_vegTypes[veg.type].height.y, c_vegTypes[veg.type].maxRadius);

		veg.radius = 0.05f * maxRadius + 0.95f * maxRadius * random::uniform_float(seed.z);
		vegBuffer[idx] = veg;
	}

	__global__ void initUniformGrid(Buffer<uint2> uniformGrid) {
		const int idx = getGlobalIndex1D();
		if (idx >= c_parameters.uniformGridCount) {
			return;
		}

		uniformGrid[idx] = { 0,0 };
	}

	// Also handles deletion of vegetation
	__global__ void fillKeys(const Buffer<Vegetation> vegBuffer, Buffer<unsigned int> keys, int count, Buffer<int> countBuffer) {
		const int idx = getGlobalIndex1D();
		if (idx >= count) {
			return;
		}

		const auto pos = vegBuffer[idx].pos;
		const int2 uniformCell = make_int2(float2{ pos.x * c_parameters.rUniformGridScale, pos.y * c_parameters.rUniformGridScale });
		const int uniformIdx = getCellIndex(getWrappedCell(uniformCell, c_parameters.uniformGridSize), c_parameters.uniformGridSize);
		constexpr unsigned int maxIndex = (unsigned int)-1;

		if (vegBuffer[idx].health < 0.f) {
			keys[idx] = maxIndex;
			atomicAdd(countBuffer, -1);
		}
		else {
			keys[idx] = uniformIdx;
		}
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
		//std::cout << count << std::endl;
		t_launchParameters.numVegetation = count;
		t_launchParameters.vegetationGridSize1D = static_cast<unsigned int>(glm::ceil(static_cast<float>(count) / static_cast<float>(t_launchParameters.blockSize1D)));
	}

	void initializeVegetation(const LaunchParameters& t_launchParameters) {
		int count = t_launchParameters.numVegetation;
		initVegetation << < t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (t_launchParameters.vegBuffer, t_launchParameters.seedBuffer, count, t_launchParameters.terrainArray);
	}

	void vegetation(LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters) {
		int count = t_launchParameters.numVegetation;
		initUniformGrid << <t_launchParameters.uniformGridSize1D, t_launchParameters.blockSize1D >> > (t_launchParameters.uniformGrid);
		fillKeys << <t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (t_launchParameters.vegBuffer, t_launchParameters.keyBuffer, count, t_launchParameters.vegetationCount);
		thrust::sort_by_key(thrust::device, t_launchParameters.keyBuffer, t_launchParameters.keyBuffer + count, t_launchParameters.vegBuffer);

		getVegetationCount(t_launchParameters);
		int diff = count - t_launchParameters.numVegetation;
		if (diff > 0) {
			std::cout << "Deleted " << count - t_launchParameters.numVegetation << " plants." << std::endl;
		}
		count = t_launchParameters.numVegetation;

		findGridStart << <t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (t_launchParameters.keyBuffer, t_launchParameters.uniformGrid, count);
		findGridEnd << <t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (t_launchParameters.keyBuffer, t_launchParameters.uniformGrid, count);


		Buffer<float> slopeBuffer{ t_launchParameters.tmpBuffer + t_simulationParameters.cellCount };

		growVegetation << <t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (t_launchParameters.vegBuffer, count, t_launchParameters.terrainArray, t_launchParameters.uniformGrid, slopeBuffer, t_launchParameters.terrainMoistureArray);
		rasterizeVegetation << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.resistanceArray, t_launchParameters.vegBuffer, t_launchParameters.uniformGrid, t_launchParameters.vegetationCount, t_launchParameters.maxVegetation, t_launchParameters.seedBuffer);
	}
}
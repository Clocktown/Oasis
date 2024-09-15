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
		const float2 height = c_parameters.vegTypeBuffer[veg.type].height;
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
		return  fmaxf(scale * expf(-0.5f * dot(d, covar * d)) - (length(float2{pos.x - veg.pos.x, pos.y - veg.pos.y}) / c_maxVegetationRadius) * expf(-2.f), 0.f); // interpolate to 0 at global max radius for uniform grid

	}

	__global__ void rasterizeVegetation(const Buffer<float> slopeBuffer)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		float4 resistance = c_parameters.resistanceArray.read(cell);
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
		const float4 terrain = c_parameters.terrainArray.read(cell);
		const float3 pos{ position.x, position.y, terrain.x + terrain.y + terrain.z };
		
		const float terrainThickness = terrain.y + terrain.z;
		const float moistureCapacityConstant = c_parameters.moistureCapacityConstant;
		const float moistureCapacity = moistureCapacityConstant * clamp(terrainThickness * c_parameters.iTerrainThicknessMoistureThreshold, 0.f, 1.f);
		const float moisture{ clamp(c_parameters.moistureArray.read(cell) / (moistureCapacity + 1e-6f), 0.f, 1.f) };

		const float slope = 2 * slopeBuffer[getCellIndex(cell)] - 1;

		float2 wind = c_parameters.windArray.read(cell);
		wind = wind / (length(wind) + 1e-6f);

		float typeProbabilities[c_maxVegTypeCount];
		float typeDensities[c_maxVegTypeCount];
		float windFactor[c_maxVegTypeCount];
		
		for (int i = 0; i < c_parameters.vegTypeCount; ++i)
		{
			typeProbabilities[i] = c_parameters.vegTypeBuffer[i].baseSpawnRate;
			typeDensities[i] = 0.f;
			windFactor[i] = 0.f;
		}
		
		const float terrainHeight = pos.z;
		float vegetationHeight = terrainHeight;

		for (int i = xStart; i <= xEnd; ++i) {
			for (int j = yStart; j <= yEnd; ++j) {
				const uint2 indices = c_parameters.uniformGrid[getCellIndex(getWrappedCell(int2{ i,j }, c_parameters.uniformGridSize), c_parameters.uniformGridSize)];
				for (unsigned int k = indices.x; k < indices.y; ++k) {
					const Vegetation veg = c_parameters.vegBuffer[k];
					const float density = getVegetationDensity(veg, pos);
					const bool isAlive = veg.health > 0.f;
					const bool canReproduce = isAlive && (veg.age > c_parameters.vegTypeBuffer[veg.type].maxMaturityTime);
					typeDensities[veg.type] += canReproduce ? density : 0.f;
					windFactor[veg.type] += canReproduce ? fmaxf(1.f - expf(-dot(wind, position - float2{ veg.pos.x, veg.pos.y })), 0.f) : 0.f;
					resistance.y += isAlive ? density : 0.f;
					resistance.w += isAlive ? c_parameters.vegTypeBuffer[veg.type].humusRate * c_parameters.deltaTime * density : density;
					vegetationHeight = fmaxf(vegetationHeight, terrainHeight + (isAlive ? density * veg.radius * c_parameters.vegTypeBuffer[veg.type].height.x : 0.f));
				}
			}
		}

		float probabilitySum = 0.f;
		for (int i = 0; i < c_parameters.vegTypeCount; ++i) {
			typeProbabilities[i] = c_parameters.vegTypeBuffer[i].baseSpawnRate * (1 + c_parameters.vegTypeBuffer[i].densitySpawnMultiplier * fminf(4.f * typeDensities[i], 1.f) + c_parameters.vegTypeBuffer[i].windSpawnMultiplier * fminf(windFactor[i], 1.f));
			const float maxRadius = fminf(fmaxf(pos.z - terrain.x, 0.f) / c_parameters.vegTypeBuffer[i].height.y, c_parameters.vegTypeBuffer[i].maxRadius);
			const bool waterCompatible = c_parameters.vegTypeBuffer[i].waterResistance >= 1.f ? terrain.w >= 0.05f * maxRadius : terrain.w * c_parameters.vegTypeBuffer[i].waterResistance <= 0.05f * maxRadius;
			const bool moistureCompatible = (moisture <= c_parameters.vegTypeBuffer[i].maxMoisture) && (moisture >= c_parameters.vegTypeBuffer[i].minMoisture);
			const bool slopeCompatible = slope <= c_parameters.vegTypeBuffer[i].maxSlope;
			const float soilCompatibility = terrain.y > 0.1f ? c_parameters.vegTypeBuffer[i].sandCompatibility * terrain.y : c_parameters.vegTypeBuffer[i].soilCompatibility * terrain.z;
			const bool soilCompatible = soilCompatibility > 0.1f;
			const float probability = (typeDensities[i] < 0.25f) && waterCompatible && moistureCompatible && slopeCompatible && soilCompatible ? typeProbabilities[i] : 0.f;
			typeProbabilities[i] = probabilitySum + probability;
			probabilitySum += probability;
		}

		// TODO: Super simplistic algorithm.
		if (*c_parameters.vegCountBuffer < c_parameters.maxVegCount) {
			int idx = getCellIndex(cell);
			uint4 seed{ c_parameters.seedBuffer[idx] };
			random::pcg(seed);
			c_parameters.seedBuffer[idx] = seed;
			const float xi = random::uniform_float(seed.x);
			const float baseProbability = c_parameters.deltaTime / c_parameters.cellCount; // TODO: Add a parameter to scale this
			if (xi < baseProbability * fmaxf(probabilitySum, 1.f)) {
				const float yi = random::uniform_float(seed.y) * fmaxf(probabilitySum, 1.f);
				Vegetation veg;
				veg.type = -1;
				for (int i = 0; i < c_parameters.vegTypeCount; ++i) {
					if (yi < typeProbabilities[i]) {
						veg.type = i;
						break;
					}
				}

				if (veg.type >= 0) {
					veg.pos = pos;
					veg.age = 0.f;
					veg.health = 1.f;
					veg.water = 0.f;
					const float maxRadius = fminf(fmaxf(veg.pos.z - terrain.x, 0.f) / c_parameters.vegTypeBuffer[veg.type].height.y, c_parameters.vegTypeBuffer[veg.type].maxRadius);

					veg.radius = 0.05f * maxRadius;
					int oldIndex = atomicAdd(c_parameters.vegCountBuffer, 1);
					if (oldIndex < c_parameters.maxVegCount) {
						c_parameters.vegBuffer[oldIndex] = veg;
					}
				}
			}
		}

		resistance.y = fminf(resistance.y, 1.f);
		c_parameters.resistanceArray.write(cell, resistance);
		c_parameters.vegHeightArray.write(cell, float2{terrainHeight, vegetationHeight});
	}

	__global__ void growVegetation(int vegCount, const Buffer<float> slopeBuffer)
	{
		const int idx = getGlobalIndex1D();
		if (idx >= vegCount) {
			return;
		}

		if (c_parameters.vegBuffer[idx].health <= 0.f || c_parameters.vegBuffer[idx].radius <= 1e-6f) {
			c_parameters.vegBuffer[idx].health = -1.f;
		}
		else {
			Vegetation veg = c_parameters.vegBuffer[idx];

			const float2 gridPos = { veg.pos.x * c_parameters.rGridScale, veg.pos.y * c_parameters.rGridScale };
			const float2 uniformGridPos = { veg.pos.x * c_parameters.rUniformGridScale, veg.pos.y * c_parameters.rUniformGridScale };
			const int xStart = int(uniformGridPos.x - 0.5f);
			const int xEnd = int(uniformGridPos.x + 0.5f);
			const int yStart = int(uniformGridPos.y - 0.5f);
			const int yEnd = int(uniformGridPos.y + 0.5f);
			const int2 cell = make_int2(gridPos); // for read from terrainArray if necessary
			const float4 terrain = c_parameters.terrainArray.read(cell);
			const float moisture = c_parameters.moistureArray.read(cell);
			const float slope = 2 * slopeBuffer[getCellIndex(cell)] - 1;
			const float bedrockHeight = terrain.x;
			const float soilHeight = bedrockHeight + terrain.z;
			const float sandHeight = soilHeight + terrain.y;
			const float waterLevel = sandHeight + terrain.w;
			const float terrainThickness = terrain.y + terrain.z;
			const float moistureCapacityConstant = c_parameters.moistureCapacityConstant;
			const float moistureCapacity = moistureCapacityConstant * clamp(terrainThickness * c_parameters.iTerrainThicknessMoistureThreshold, 0.f, 1.f);
			const float relativeMoisture{ clamp(moisture / (moistureCapacity + 1e-6f), 0.f, 1.f) };
			const float2 shadowHeights = c_parameters.vegHeightArray.read(cell);
			const float2 shadow = c_parameters.shadowArray.read(cell);

			float overlap = 0.f;

			for (int i = xStart; i <= xEnd; ++i) {
				for (int j = yStart; j <= yEnd; ++j) {
					const uint2 indices = c_parameters.uniformGrid[getCellIndex(getWrappedCell(int2{ i,j }, c_parameters.uniformGridSize), c_parameters.uniformGridSize)];
					for (unsigned int k = indices.x; k < indices.y; ++k) {
						if (k == idx) continue;
						// TODO: Very adhoc formula. Doesn't really consider the volume. Read literature what they actually do.
						const float incompatibility = c_parameters.vegMatrixBuffer[getCellIndex(int2{ c_parameters.vegBuffer[k].type, veg.type }, int2{ c_maxVegTypeCount })];
						float d = -0.5f * fminf(length(c_parameters.vegBuffer[k].pos - veg.pos) - (veg.radius + c_parameters.vegBuffer[k].radius), 0); // TODO: pos is not centered right now. Maybe have it centered and change height? or compute center here?
						d /= veg.radius;
						overlap += incompatibility * d * d;
					}
				}
			}
			// Age plant
			veg.age += c_parameters.deltaTime;

			veg.pos.z += c_parameters.vegTypeBuffer[veg.type].positionAdjustRate * (sandHeight - veg.pos.z);

			// Plant parameters
			const float plantHeight = (veg.radius * c_parameters.vegTypeBuffer[veg.type].height.x);
			const float plantDepth = (veg.radius * c_parameters.vegTypeBuffer[veg.type].height.y);
			const bool isWaterPlant = c_parameters.vegTypeBuffer[veg.type].waterResistance >= 1.f;

			// Soil/Sand root conditions
			const float plantBottom = veg.pos.z - plantDepth;
			const float plantTop = veg.pos.z + plantHeight;
			const float soilCoverage = clamp((fminf(soilHeight, veg.pos.z) - fmaxf(plantBottom, bedrockHeight)) / plantDepth, 0.f, 1.f);
			const float sandCoverage = clamp((fminf(sandHeight, veg.pos.z) - fmaxf(plantBottom, soilHeight)) / plantDepth, 0.f, 1.f);
			const float soilRate = soilCoverage * c_parameters.vegTypeBuffer[veg.type].soilCompatibility + sandCoverage * c_parameters.vegTypeBuffer[veg.type].sandCompatibility;
			const float rootCoverage = clamp((fminf(sandHeight, veg.pos.z) - fmaxf(plantBottom, bedrockHeight)) / plantDepth, 0.f, 1.f);
			const float stemCoverage = clamp((fminf(sandHeight, plantTop) - fmaxf(veg.pos.z, bedrockHeight)) / plantHeight, 0.f, 1.f);
			const float rootDamage = -(rootCoverage - c_parameters.vegTypeBuffer[veg.type].terrainCoverageResistance.x) / c_parameters.vegTypeBuffer[veg.type].terrainCoverageResistance.x;
			const float stemDamage = (stemCoverage - c_parameters.vegTypeBuffer[veg.type].terrainCoverageResistance.y) / (1 - c_parameters.vegTypeBuffer[veg.type].terrainCoverageResistance.y);

			// Water usage
			const float waterCapacity = 4.1888f * (plantDepth + plantHeight) * veg.radius * veg.radius * c_parameters.vegTypeBuffer[veg.type].waterStorageCapacity;
			const float availableGroundWater = c_parameters.deltaTime * 4.1888f * plantDepth * veg.radius * veg.radius * moisture;
			const float requiredWater = c_parameters.vegTypeBuffer[veg.type].waterUsageRate * c_parameters.deltaTime * 4.1888f * plantHeight * veg.radius * veg.radius;
			const float waterDifference = availableGroundWater - requiredWater;
			const float missingWater = clamp(((availableGroundWater + veg.water) - requiredWater) / requiredWater, -1.f, 1.f);
			if (waterDifference > 0.f) {
				veg.water = fminf(veg.water + c_parameters.deltaTime * fmaxf(waterCapacity - veg.water, 0.f) * waterDifference, waterCapacity);
			}
			else if (waterDifference < 0.f) {
				veg.water = fmaxf(veg.water + waterDifference, 0.f);
			}
			const float thirstDamage = fmaxf(-missingWater, 0.f);
			const float thirstGrowth = 1.f + missingWater;

			// Moisture compatibility
			const float moistureDamage = fmaxf((relativeMoisture - c_parameters.vegTypeBuffer[veg.type].maxMoisture) / (1.f - c_parameters.vegTypeBuffer[veg.type].maxMoisture), 0.f);
			const float moistureGrowth = moistureDamage > 0.f ? 0.f : 1.f;

			// Slope compatibility
			const float slopeDamage = fmaxf((slope - c_parameters.vegTypeBuffer[veg.type].maxSlope) / (1.f - c_parameters.vegTypeBuffer[veg.type].maxSlope), 0.f);
			const float slopeGrowth = fmaxf((c_parameters.vegTypeBuffer[veg.type].maxSlope - slope) / c_parameters.vegTypeBuffer[veg.type].maxSlope, 0.f);
			 
			// Surface water conditions
			const float waterOverlap = clamp((waterLevel - fmaxf(veg.pos.z, sandHeight)) / plantHeight, 0.f, 1.f);
			const float waterRate = (waterOverlap - c_parameters.vegTypeBuffer[veg.type].waterResistance);
			const float waterDamage = isWaterPlant ? 0.f : waterRate / (1.f - c_parameters.vegTypeBuffer[veg.type].waterResistance);
			const float waterGrowth = isWaterPlant ? 1.f : fmaxf(-waterRate / c_parameters.vegTypeBuffer[veg.type].waterResistance, 0.f);

			// Light conditions
			const float plantShadowHeight = shadowHeights.x + plantHeight;
			const float heightDiff = (shadowHeights.y - plantShadowHeight);
			const float shadow_t = (shadowHeights.y - shadowHeights.x) > 1e-6f ? clamp(heightDiff / (shadowHeights.y - shadowHeights.x), 0.f, 1.f) : 1.f;
			const float shadowVal = lerp(shadow.y, shadow.x, shadow_t) * (isWaterPlant ? expf(-0.1f*terrain.w) : 1.f);
			const float lightIntervalMax = 0.5f * (c_parameters.vegTypeBuffer[veg.type].lightConditions.x + c_parameters.vegTypeBuffer[veg.type].lightConditions.y);
			const float shadowGrowth = 1.f - 2.f * abs(shadowVal - lightIntervalMax) / (c_parameters.vegTypeBuffer[veg.type].lightConditions.y - c_parameters.vegTypeBuffer[veg.type].lightConditions.x);

			// Max radius based on neighborhood and terrain
			const float baseMaxRadius = fmaxf(1.f - overlap, 0.f) * fmaxf(shadowGrowth, 0.f) * fminf(fmaxf(veg.pos.z - bedrockHeight, 0.f) / c_parameters.vegTypeBuffer[veg.type].height.y, c_parameters.vegTypeBuffer[veg.type].maxRadius);
			const float waterMaxRadius = fminf(baseMaxRadius, fmaxf((waterLevel - veg.pos.z) / c_parameters.vegTypeBuffer[veg.type].height.x, 0.f));
			const float maxRadius = isWaterPlant ? waterMaxRadius : baseMaxRadius;

			// 1.1 * maxRadius serves as a Buffer and prevents oscillations
			// Shrink, if possible
			veg.radius -= clamp(c_parameters.vegTypeBuffer[veg.type].shrinkRate * c_parameters.deltaTime * (veg.radius > 1.1*maxRadius ? 1.f : 0.f), 0.f, veg.radius - maxRadius);

			// Damage plant if it is still too large given the current conditions
			const float radiusDamage = fmaxf((veg.radius - 1.1*maxRadius) / veg.radius, 0.f);
			const float radiusRate = radiusDamage > 0.f ? 0.f : 1.f;

			// Calculate growth and health
			const float growthRate = fmaxf(shadowGrowth, 0.f) * radiusRate * slopeGrowth * moistureGrowth * thirstGrowth * soilRate * waterGrowth * fmaxf(1.f - overlap, 0.f) * c_parameters.deltaTime * c_parameters.vegTypeBuffer[veg.type].growthRate;
			veg.health -= c_parameters.vegTypeBuffer[veg.type].damageRate * c_parameters.deltaTime * (fmaxf(-shadowGrowth, 0.f) + fmaxf(waterDamage, 0.f) + fmaxf(rootDamage, 0.f) + fmaxf(stemDamage, 0.f) + thirstDamage + moistureDamage + slopeDamage + radiusDamage);
			// TODO: maybe also a constant rate?
			veg.health += growthRate;

			// Calculate maximum radius based on root depth and bedrock and grow plant
			const float newRadius = fminf(veg.radius + growthRate, maxRadius);
			veg.radius = newRadius > veg.radius ? newRadius : veg.radius;

			// Check if maturity condition met
			if (veg.age > c_parameters.vegTypeBuffer[veg.type].maxMaturityTime && veg.radius < c_parameters.vegTypeBuffer[veg.type].maturityPercentage * c_parameters.vegTypeBuffer[veg.type].maxRadius) {
				veg.health = 0.f;
			}
			veg.health = clamp(veg.health, 0.f, 1.f);
			c_parameters.vegBuffer[idx] = veg;
		}
	}


	// temporary random fill
	__global__ void initVegetation(int vegCount) {
		const int idx = getGlobalIndex1D();
		if (idx >= vegCount) {
			return;
		}

		uint4 seed = c_parameters.seedBuffer[idx];
		random::pcg(seed);
		c_parameters.seedBuffer[idx] = seed;

		Vegetation veg;
		veg.type = seed.w % 2;
		veg.age = 0.f;
		veg.health = 1.f;
		veg.water = 0.f;
		const int2 vegCell{ seed.x % c_parameters.gridSize.x, seed.y % c_parameters.gridSize.y };
		const float4 terrain = c_parameters.terrainArray.read(vegCell);
		veg.pos = { (vegCell.x + 0.5f) * c_parameters.gridScale, (vegCell.y + 0.5f) * c_parameters.gridScale, terrain.x + terrain.y + terrain.z };
		const float maxRadius = fminf(fmaxf(veg.pos.z - terrain.x, 0.f) / c_parameters.vegTypeBuffer[veg.type].height.y, c_parameters.vegTypeBuffer[veg.type].maxRadius);

		veg.radius = 0.05f * maxRadius + 0.95f * maxRadius * random::uniform_float(seed.z);
		c_parameters.vegBuffer[idx] = veg;
	}

	__global__ void initUniformGrid() {
		const int idx = getGlobalIndex1D();
		if (idx >= c_parameters.uniformGridCount) {
			return;
		}

		c_parameters.uniformGrid[idx] = { 0,0 };
	}

	// Also handles deletion of vegetation
	__global__ void fillKeys(int count) {
		const int idx = getGlobalIndex1D();
		if (idx >= count) {
			return;
		}

		const auto pos = c_parameters.vegBuffer[idx].pos;
		const int2 uniformCell = make_int2(float2{ pos.x * c_parameters.rUniformGridScale, pos.y * c_parameters.rUniformGridScale });
		const int uniformIdx = getCellIndex(getWrappedCell(uniformCell, c_parameters.uniformGridSize), c_parameters.uniformGridSize);
		constexpr unsigned int maxIndex = (unsigned int)-1;

		if (c_parameters.vegBuffer[idx].health < 0.f) {
			c_parameters.keyBuffer[idx] = maxIndex;
			atomicAdd(c_parameters.vegCountBuffer, -1);
		}
		else {
			c_parameters.keyBuffer[idx] = uniformIdx;
		}
	}

	__global__ void findGridStart(int count) {
		const int idx = getGlobalIndex1D();
		if (idx >= count) {
			return;
		}

		const bool isZero = idx == 0;
		const unsigned int uniformIdxA = c_parameters.keyBuffer[idx];
		const unsigned int uniformIdxB = isZero ? 0 : c_parameters.keyBuffer[idx - 1];

		if (isZero || uniformIdxB != uniformIdxA) {
			c_parameters.uniformGrid[uniformIdxA].x = idx;
		}
	}

	__global__ void findGridEnd(int count) {
		const int idx = getGlobalIndex1D();
		if (idx >= count) {
			return;
		}

		const bool isLast = idx == (count - 1);
		const unsigned int uniformIdxA = c_parameters.keyBuffer[idx];
		const unsigned int uniformIdxB = isLast ? 0 : c_parameters.keyBuffer[idx + 1];

		if (isLast || uniformIdxB != uniformIdxA) {
			c_parameters.uniformGrid[uniformIdxA].y = idx + 1;
		}
	}

	__global__ void prepareVegMapKernel(Buffer<int> relMapBuffer)
	{
		const int count{ *c_parameters.vegCountBuffer };
		const int stride{ getGridStride1D() };

		for (int index{ getGlobalIndex1D() }; index < count; index += stride)
		{
			relMapBuffer[index] = atomicAdd(c_parameters.vegCountBuffer + 1 + c_parameters.vegBuffer[index].type, 1);
		}
	}

	__global__ void vegMapKernel(Buffer<int> relMapBuffer)
	{
		const int count{ *c_parameters.vegCountBuffer };
		const int stride{ getGridStride1D() };

		int offsets[c_maxVegTypeCount];
		offsets[0] = 0;

		for (int i{ 1 }; i < c_parameters.vegTypeCount; ++i)
		{
			offsets[i] = offsets[i - 1] + c_parameters.vegCountBuffer[1 + (i - 1)];
		}

		for (int index{ getGlobalIndex1D() }; index < count; index += stride)
		{
			c_parameters.vegMapBuffer[offsets[c_parameters.vegBuffer[index].type] + relMapBuffer[index]] = index;
		}
	}

	__global__ void calculateShadowMap() {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const float2 heights{ c_parameters.vegHeightArray.read(cell) }; // .x is height without Veg, .y is height with Veg
		float2 shadow{ 0.f, 0.f }; // .x is shadow at ground level, .y is shadow at top of vegetation
		const float2 lightDirection = { -sqrtf(2.f), -sqrtf(2.f) };

		const float2 position{ make_float2(cell) + 0.5f };

		for (int i = -3; i <= 3; ++i) {
			for (int j = -3; j <= 3; ++j) {
				const float2 offset = make_float2(int2{ i, j });
				const float distance = length(offset) * c_parameters.gridScale;
				float2 nextPosition = position - offset;

				const float nextHeight{ c_parameters.vegHeightArray.sample(nextPosition).y }; // .y is the height including Vegetation
				const float2 heightsDifference{ nextHeight - heights.x, nextHeight - heights.y };
				const float2 angle{ heightsDifference.x / distance, heightsDifference.y / distance };

				const float d = -2.f * fmaxf(dot(offset, lightDirection) * c_parameters.gridScale, 0.f) * pow(distance, -2.f);

				shadow += float2{ 
					clamp(expf(d * angle.x), 0.f, 1.f),
					clamp(expf(d * angle.y), 0.f, 1.f)
				};
			}
		}

		shadow *= (1.f / 49.f);

		c_parameters.shadowArray.write(cell, clamp(2.f * (shadow - 0.5f), 0, 1)); // Rescaling shadow, because the dot product means that shadow can't be smaller than 0.5 (roughly)
	}

	void getVegetationCount(LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters) {
		int counts[1 + c_maxVegTypeCount];
		cudaMemcpy(counts, t_simulationParameters.vegCountBuffer, (1 + c_maxVegTypeCount) * sizeof(int), cudaMemcpyDeviceToHost);
		
		if (counts[0] > t_launchParameters.maxVegCount)
		{
			counts[0] = t_launchParameters.maxVegCount;
			cudaMemcpy(t_simulationParameters.vegCountBuffer, counts, sizeof(int), cudaMemcpyHostToDevice);
		}

		t_launchParameters.vegCount = counts[0];

		for (int i{ 0 }; i < c_maxVegTypeCount; ++i) 
		{
			t_launchParameters.vegCountsPerType[i] = counts[1 + i];
		}

		t_launchParameters.vegetationGridSize1D = counts[0] == 0 ? 1 : static_cast<unsigned int>(glm::ceil(static_cast<float>(counts[0]) / static_cast<float>(t_launchParameters.blockSize1D)));
	}

	void initializeVegetation(const LaunchParameters& t_launchParameters) {
		int count = t_launchParameters.vegCount;
		initVegetation << < t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (count);
	}

	void vegetation(LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters) {
		int count = t_launchParameters.vegCount;
		initUniformGrid << <t_launchParameters.uniformGridSize1D, t_launchParameters.blockSize1D >> > ();
		
		Buffer<float> slopeBuffer{ t_launchParameters.tmpBuffer + t_simulationParameters.cellCount };
		Buffer<int> relMapBuffer{ reinterpret_cast<Buffer<int>>(slopeBuffer + t_simulationParameters.cellCount) };

		if (count > 0) {
			fillKeys<<<t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (count);
			thrust::sort_by_key(thrust::device, t_simulationParameters.keyBuffer, t_simulationParameters.keyBuffer + count, t_simulationParameters.vegBuffer);

			// memset type counter 0
			CU_CHECK_ERROR(cudaMemset(t_simulationParameters.vegCountBuffer + 1, 0, c_maxVegTypeCount * sizeof(int)));

			// atomic adds for type counter
			// return of atomic in tmp buffer for veg id
			prepareVegMapKernel<<<t_launchParameters.optimalGridSize1D, t_launchParameters.optimalBlockSize1D>>>(relMapBuffer);
			
			// second pass use relative offset in tmp buffer + global counter offset to write global id in map 
			vegMapKernel<<<t_launchParameters.optimalGridSize1D, t_launchParameters.optimalBlockSize1D>>>(relMapBuffer);

			getVegetationCount(t_launchParameters, t_simulationParameters);
			int diff = count - t_launchParameters.vegCount;
			if (diff > 0) {
				//std::cout << "Deleted " << count - t_launchParameters.numVegetation << " plants." << std::endl;
			}
			count = t_launchParameters.vegCount;

			findGridStart<<<t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (count);
			findGridEnd<<<t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (count);
		}

		// Fix double humus generation
		growVegetation<<<t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (count, slopeBuffer);
		rasterizeVegetation<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (slopeBuffer);
		calculateShadowMap << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > ();
	}
}
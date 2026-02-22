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
#include <sthe/cu/stopwatch.hpp>
#include <cuda_fp16.h>

#include "random.cuh"

namespace dunes {

	__forceinline__ __device__ float getVegetationDensity(const Vegetation& veg, const float3& pos) {
        const float  r2     = 0.25f * __half2float(veg.pos_r.b.y) * __half2float(veg.pos_r.b.y);
		const float2 height = __half22float2((*c_parameters.vegTypeBuffer).height[veg.type]);
		const float stem2 = height.x * height.x * r2;
		const float root2 = height.y * height.y * r2;
		const float3 covarStem{ 1.f / r2, 1.f / r2, 1.f / stem2 };
		const float3 covarRoot{ covarStem.x, covarStem.y, 1.f / root2 };
		const float3 covar = pos.z >= __half2float(veg.pos_r.b.x) ? covarStem : covarRoot;

		// Compute distance vector while considering toroidal boundaries
        float2 pxy = __half22float2(veg.pos_r.a);
		float3 d = abs(pos - float3(pxy.x, pxy.y, __half2float(veg.pos_r.b.x)));
		const float2 dims{ make_float2(c_parameters.gridSize) * c_parameters.gridScale };
		d.x = fminf(d.x, fabs(d.x - dims.x));
		d.y = fminf(d.y, fabs(d.y - dims.y));

		const float scale = 1.f; // peak vegetation density
		// gaussian distribution faded toward 0 at c_maxVegetationRadius
		return  fmaxf(scale * expf(-0.5f * dot(d, covar * d)) - (length(float2{pos.x - pxy.x, pos.y - pxy.y}) / c_maxVegetationRadius) * expf(-2.f), 0.f); // interpolate to 0 at global max radius for uniform grid

	}

	__global__ void rasterizeVegetation(const Buffer<half> slopeBuffer, int count)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		float4 resistance = half4toFloat4(c_parameters.resistanceArray.read(cell));

		if (resistance.y < 0.f) 
		{
			return;
		}

		resistance.y = 0.f;

		const float2 position{ (make_float2(cell) + 0.5f) * c_parameters.gridScale };
		const float4 terrain = half4toFloat4(c_parameters.terrainArray.read(cell));
		const float3 pos{ position.x, position.y, terrain.x + terrain.y + terrain.z };
		
		const float terrainThickness = terrain.y + terrain.z;
		const float moistureCapacityConstant = c_parameters.moistureCapacityConstant;
		const float moistureCapacity = moistureCapacityConstant * clamp(terrainThickness * c_parameters.iTerrainThicknessMoistureThreshold, 0.f, 1.f);
		const float moisture{ clamp(__half2float(c_parameters.moistureArray.read(cell)) / (moistureCapacity + 1e-6f), 0.f, 1.f) };

		const float slope = 2.f * __half2float(slopeBuffer[getCellIndex(cell)]) - 1.f;

		float2 wind = sampleLinearOrNearest<true>(c_parameters.windArray, 0.5f * (make_float2(cell) + 0.5f));
		wind = wind / (length(wind) + 1e-6f);

		float typeProbabilities[c_maxVegTypeCount];
		float typeDensities[c_maxVegTypeCount];
		float windFactor[c_maxVegTypeCount];
		
		for (int i = 0; i < c_parameters.vegTypeCount; ++i)
		{
			typeDensities[i] = 0.f;
			windFactor[i] = 0.f;
		}
		
		const float terrainHeight = pos.z;
		float vegetationHeight = terrainHeight;

		// Adaptive Grid
		for (int l = 0; l < c_parameters.adaptiveGrid.layerCount; ++l)
		{
			const float2 gridPosition{ position / c_parameters.adaptiveGrid.gridScales[l] };
			const int xStart = int(gridPosition.x - 0.5f);
			const int xEnd = int(gridPosition.x + 0.5f);
			const int yStart = int(gridPosition.y - 0.5f);
			const int yEnd = int(gridPosition.y + 0.5f);

			for (int i = xStart; i <= xEnd; ++i) {
				for (int j = yStart; j <= yEnd; ++j) {
					const int gridCellIndex = getCellIndex(getWrappedCell(int2{ i,j }, c_parameters.adaptiveGrid.gridSizes[l]), c_parameters.adaptiveGrid.gridSizes[l]);
					const unsigned int idxStart = c_parameters.adaptiveGrid.gridBuffer[l][gridCellIndex];
					const unsigned int idxEnd = c_parameters.adaptiveGrid.gridBuffer[l][gridCellIndex + 1];
					if (idxEnd == 0xFFFFFFFF) continue;
					for (unsigned int k = idxStart; k < idxEnd; ++k) {
						const Vegetation veg = c_parameters.vegBuffer[k];
						const float density = getVegetationDensity(veg, pos);
						const bool isAlive = veg.health > CUDART_ZERO_FP16;
						const bool canReproduce = isAlive && (veg.age > (*c_parameters.vegTypeBuffer).maxMaturityTime[veg.type]);
						typeDensities[veg.type] += canReproduce ? density : 0.f;
						const float2 off = position - __half22float2(veg.pos_r.a);
						const float dist = length(off) + 1e-6f;
						windFactor[veg.type] += canReproduce ? lerp(fmaxf(dot(wind, off/dist), 0.f), 0.f, fminf(dist / __half2float((*c_parameters.vegTypeBuffer).maxRadius[veg.type]), 1.f))/*fmaxf(1.f - expf(-dot(wind, position - float2{veg.pos.x, veg.pos.y})), 0.f)*/ : 0.f;
						resistance.y += isAlive ? density : 0.f;
						resistance.w += isAlive ? __half2float((*c_parameters.vegTypeBuffer).humusRate[veg.type]) * c_parameters.deltaTime * density : density;
						vegetationHeight = fmaxf(vegetationHeight, terrainHeight + (isAlive ? density * __half2float(veg.pos_r.b.y) * __half2float((*c_parameters.vegTypeBuffer).height[veg.type].x) : 0.f));
					}
				}
			}
		}

		float probabilitySum = 0.f;
		for (int i = 0; i < c_parameters.vegTypeCount; ++i) {
			typeProbabilities[i] = 
				__half2float((*c_parameters.vegTypeBuffer).baseSpawnRate[i]) * (1 + __half2float((*c_parameters.vegTypeBuffer).densitySpawnMultiplier[i]) * 
					fminf(typeDensities[i], 1.f) + __half2float((*c_parameters.vegTypeBuffer).windSpawnMultiplier[i]) * fminf(windFactor[i], 1.f));
			const float maxRadius = fminf(fmaxf(pos.z - terrain.x, 0.f) / __half2float((*c_parameters.vegTypeBuffer).height[i].y), __half2float((*c_parameters.vegTypeBuffer).maxRadius[i]));
			const bool waterCompatible = (*c_parameters.vegTypeBuffer).waterResistance[i] >= CUDART_ONE_FP16 ? (terrain.w >= 0.05f * maxRadius * __half2float((*c_parameters.vegTypeBuffer).height[i].x)) : (terrain.w <= __half2float((*c_parameters.vegTypeBuffer).waterResistance[i]) * 0.05f * maxRadius * __half2float((*c_parameters.vegTypeBuffer).height[i].x));
			const bool moistureCompatible = (moisture <= __half2float((*c_parameters.vegTypeBuffer).maxMoisture[i])) && (moisture >= __half2float((*c_parameters.vegTypeBuffer).minMoisture[i]));
			const bool slopeCompatible = slope <= __half2float((*c_parameters.vegTypeBuffer).maxSlope[i]);
			const float soilCompatibility = terrain.y > 0.1f ? __half2float((*c_parameters.vegTypeBuffer).sandCompatibility[i]) * terrain.y : __half2float((*c_parameters.vegTypeBuffer).soilCompatibility[i]) * terrain.z;
			const bool soilCompatible = soilCompatibility > 0.1f;
			const float probability = (typeDensities[i] < __half2float((*c_parameters.vegTypeBuffer).separation[i])) && waterCompatible && moistureCompatible && slopeCompatible && soilCompatible ? typeProbabilities[i] : 0.f;
			typeProbabilities[i] = probabilitySum + probability;
			probabilitySum += probability;
		}

		
		if (*c_parameters.vegCountBuffer < c_parameters.maxVegCount) {
			int idx = getCellIndex(cell);
			uint4 seed{ c_parameters.seedBuffer[idx] };
			random::pcg(seed);
			c_parameters.seedBuffer[idx] = seed;
			const float xi = random::uniform_float(seed.x);
			const float baseProbability = 1.f - pow(1.f - fminf(probabilitySum / c_parameters.cellCount, 1.f), c_parameters.deltaTime);
			if (xi < baseProbability) {
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
                    const float maxRadius =
                            fminf(fmaxf(pos.z - terrain.x, 0.f) /
                                            __half2float((*c_parameters.vegTypeBuffer)
                                                                .height[veg.type]
                                                                .y),
                                    __half2float((*c_parameters.vegTypeBuffer)
                                                        .maxRadius[veg.type]));

					veg.pos_r = toHalf4({pos.x, pos.y, pos.z, 0.05f * maxRadius});
					veg.age = CUDART_ZERO_FP16;
					veg.health = CUDART_ONE_FP16;
					veg.water = CUDART_ZERO_FP16;
					int oldIndex = atomicAdd(c_parameters.vegCountBuffer, 1);
					if (oldIndex < c_parameters.maxVegCount) {
						c_parameters.vegBuffer[oldIndex] = veg;
					}
				}
			}
		}

		resistance.y = fminf(resistance.y, 1.f);
		c_parameters.resistanceArray.write(cell, toHalf4(resistance));
		c_parameters.vegHeightArray.write(cell, __float22half2_rn(float2{terrainHeight, vegetationHeight}));
	}

	__global__ void growVegetation(int vegCount, const Buffer<half> slopeBuffer)
	{
		const int idx = getGlobalIndex1D();
		if (idx >= vegCount) {
			return;
		}

		if (c_parameters.vegBuffer[idx].health <= CUDART_ZERO_FP16 || __half2float(c_parameters.vegBuffer[idx].pos_r.b.y) <= 1e-6f) {
			c_parameters.vegBuffer[idx].health = - CUDART_ONE_FP16;
		}
		else {
			Vegetation veg = c_parameters.vegBuffer[idx];

			const float  vegRad  = __half2float(veg.pos_r.b.y);
			const float2 vegPos  = __half22float2(veg.pos_r.a);
			const float2 gridPos = { vegPos * c_parameters.rGridScale };
			const int2 cell = make_int2(gridPos); // for read from terrainArray if necessary
			const float4 terrain = half4toFloat4(c_parameters.terrainArray.read(cell));
			const float moisture = __half2float(c_parameters.moistureArray.read(cell));
			const float slope = 2.f * __half2float(slopeBuffer[getCellIndex(cell)]) - 1.f;
			const float bedrockHeight = terrain.x;
			const float soilHeight = bedrockHeight + terrain.z;
			const float sandHeight = soilHeight + terrain.y;
			const float waterLevel = sandHeight + terrain.w;
			const float terrainThickness = terrain.y + terrain.z;
			const float moistureCapacityConstant = c_parameters.moistureCapacityConstant;
			const float moistureCapacity = moistureCapacityConstant * clamp(terrainThickness * c_parameters.iTerrainThicknessMoistureThreshold, 0.f, 1.f);
			const float relativeMoisture{ clamp(moisture / (moistureCapacity + 1e-6f), 0.f, 1.f) };
			const float2 shadowHeights = __half22float2(c_parameters.vegHeightArray.read(cell));
			const float2 shadow = __half22float2(c_parameters.shadowArray.read(cell));

			float overlap = 0.f;

			for (int l = 0; l < c_parameters.adaptiveGrid.layerCount; ++l)
			{
				const int xStart = int(floorf((vegPos.x - vegRad - 0.5f * c_parameters.adaptiveGrid.gridScales[l]) / c_parameters.adaptiveGrid.gridScales[l]));
				const int xEnd = int(ceilf((vegPos.x + vegRad + 0.5f * c_parameters.adaptiveGrid.gridScales[l]) / c_parameters.adaptiveGrid.gridScales[l]));
				const int yStart = int(floorf((vegPos.y - vegRad - 0.5f * c_parameters.adaptiveGrid.gridScales[l]) / c_parameters.adaptiveGrid.gridScales[l]));
				const int yEnd = int(ceilf((vegPos.y + vegRad + 0.5f * c_parameters.adaptiveGrid.gridScales[l]) / c_parameters.adaptiveGrid.gridScales[l]));

				for (int i = xStart; i <= xEnd; ++i) {
					for (int j = yStart; j <= yEnd; ++j) {
						const int gridCellIndex = getCellIndex(getWrappedCell(int2{ i,j }, c_parameters.adaptiveGrid.gridSizes[l]), c_parameters.adaptiveGrid.gridSizes[l]);
						const unsigned int idxStart = c_parameters.adaptiveGrid.gridBuffer[l][gridCellIndex];
						const unsigned int idxEnd = c_parameters.adaptiveGrid.gridBuffer[l][gridCellIndex + 1];
						if (idxEnd == 0xFFFFFFFF) continue;
						for (unsigned int k = idxStart; k < idxEnd; ++k) {
							if (k == idx) continue;
							const float incompatibility = c_parameters.vegMatrixBuffer[getCellIndex(int2{ c_parameters.vegBuffer[k].type, veg.type }, int2{ c_maxVegTypeCount })];

							float d =
                                    -0.5f *
                                    fminf(length(float3 {__half2float(c_parameters.vegBuffer[k].pos_r.a.x),
                                                            __half2float(c_parameters.vegBuffer[k].pos_r.a.y),
                                                            __half2float(c_parameters.vegBuffer[k].pos_r.b.x)
										} -
                                                    float3 {vegPos.x,
                                                            vegPos.y,
                                                            __half2float(veg.pos_r.b.x)}) -
                                                    (vegRad +
                                                    __half2float(
                                                            c_parameters
                                                                    .vegBuffer
                                                                            [k]
                                                                    .pos_r
                                                                    .b
                                                                    .y)),
                                            0.f);
							d /= vegRad;
							overlap += incompatibility * d * d;
						}
					}
				}
			}
			// Age plant
			veg.age += __float2half(c_parameters.deltaTime);

			veg.pos_r.b.x += __float2half(__half2float((*c_parameters.vegTypeBuffer).positionAdjustRate[veg.type]) *
                                (sandHeight - __half2float(veg.pos_r.b.x)));

			// Plant parameters
			const float plantHeight = (vegRad * __half2float((*c_parameters.vegTypeBuffer).height[veg.type].x));
			const float plantDepth = (vegRad * __half2float((*c_parameters.vegTypeBuffer).height[veg.type].y));
			const bool isWaterPlant = (*c_parameters.vegTypeBuffer).waterResistance[veg.type] >= CUDART_ONE_FP16;

			// Soil/Sand root conditions
            const float plantBottom = __half2float(veg.pos_r.b.x) - plantDepth;
            const float plantTop     = __half2float(veg.pos_r.b.x) + plantHeight;
            const float soilCoverage = clamp((fminf(soilHeight, __half2float(veg.pos_r.b.x)) -
                                              fmaxf(plantBottom, bedrockHeight)) /
                                                     plantDepth,
                                             0.f,
                                             1.f);
            const float sandCoverage = clamp((fminf(sandHeight, __half2float(veg.pos_r.b.x)) -
                                              fmaxf(plantBottom, soilHeight)) /
                                                     plantDepth,
                                             0.f,
                                             1.f);
			const float soilRate = soilCoverage * __half2float((*c_parameters.vegTypeBuffer).soilCompatibility[veg.type]) + sandCoverage * __half2float((*c_parameters.vegTypeBuffer).sandCompatibility[veg.type]);
            const float rootCoverage = clamp((fminf(sandHeight, __half2float(veg.pos_r.b.x)) -
                                              fmaxf(plantBottom, bedrockHeight)) /
                                                     plantDepth,
                                             0.f,
                                             1.f);
                        const float stemCoverage =
                                clamp((fminf(sandHeight, plantTop) -
                                       fmaxf(__half2float(veg.pos_r.b.x), bedrockHeight)) /
                                              plantHeight,
                                      0.f,
                                      1.f);
			const float rootDamage = -(rootCoverage - __half2float((*c_parameters.vegTypeBuffer).terrainCoverageResistance[veg.type].x)) / __half2float((*c_parameters.vegTypeBuffer).terrainCoverageResistance[veg.type].x);
			const float stemDamage = (stemCoverage - __half2float((*c_parameters.vegTypeBuffer).terrainCoverageResistance[veg.type].y)) / (1.f - __half2float((*c_parameters.vegTypeBuffer).terrainCoverageResistance[veg.type].y));

			// Water usage
			const float waterCapacity = 2.094f * (plantDepth + plantHeight) * vegRad * vegRad * __half2float((*c_parameters.vegTypeBuffer).waterStorageCapacity[veg.type]);
			const float availableGroundWater = c_parameters.deltaTime * 2.094f * plantDepth * vegRad * vegRad * moisture;
			const float requiredWater = __half2float((*c_parameters.vegTypeBuffer).waterUsageRate[veg.type]) * c_parameters.deltaTime * 2.094f * plantHeight * vegRad * vegRad;
			const float waterDifference = availableGroundWater - requiredWater;
			const float missingWater = clamp(((availableGroundWater + __half2float(veg.water)) - requiredWater) / requiredWater, -1.f, 1.f);
			if (waterDifference > 0.f) {
                            veg.water =
                                    __float2half(fminf(__half2float(veg.water) +
                                                  c_parameters.deltaTime *
                                                    fmaxf(waterCapacity - __half2float(veg.water),
                                                          0.f) *
                                                          waterDifference,
                                          waterCapacity));
			}
			else if (waterDifference < 0.f) {
                            veg.water = __float2half(fmaxf(__half2float(veg.water) + waterDifference, 0.f));
			}
			const float thirstDamage = fmaxf(-missingWater, 0.f);
			const float thirstGrowth = 1.f + missingWater;

			// Moisture compatibility
			const float moistureDamage = fmaxf((relativeMoisture - __half2float((*c_parameters.vegTypeBuffer).maxMoisture[veg.type])) / (1.f - __half2float((*c_parameters.vegTypeBuffer).maxMoisture[veg.type])), 0.f);
			const float moistureGrowth = moistureDamage > 0.f ? 0.f : 1.f;

			// Slope compatibility
			const float slopeDamage = fmaxf((slope - __half2float((*c_parameters.vegTypeBuffer).maxSlope[veg.type])) / (1.f - __half2float((*c_parameters.vegTypeBuffer).maxSlope[veg.type])), 0.f);
			const float slopeGrowth = fmaxf((__half2float((*c_parameters.vegTypeBuffer).maxSlope[veg.type]) - slope) / __half2float((*c_parameters.vegTypeBuffer).maxSlope[veg.type]), 0.f);
			 
			// Surface water conditions
                        const float waterOverlap = clamp(
                                (waterLevel - fmaxf(__half2float(veg.pos_r.b.x), sandHeight)) /
                                        plantHeight,
                                0.f,
                                1.f);
			const float waterRate = (waterOverlap - __half2float((*c_parameters.vegTypeBuffer).waterResistance[veg.type]));
			const float waterDamage = isWaterPlant ? 0.f : waterRate / (1.f - __half2float((*c_parameters.vegTypeBuffer).waterResistance[veg.type]));
			const float waterGrowth = isWaterPlant ? 1.f : fmaxf(-waterRate / __half2float((*c_parameters.vegTypeBuffer).waterResistance[veg.type]), 0.f);

			// Light conditions
			const float plantShadowHeight = shadowHeights.x + plantHeight;
			const float heightDiff = (shadowHeights.y - plantShadowHeight);
			const float shadow_t = (shadowHeights.y - shadowHeights.x) > 1e-6f ? clamp(heightDiff / (shadowHeights.y - shadowHeights.x), 0.f, 1.f) : 1.f;
			const float shadowVal = lerp(shadow.y, shadow.x, shadow_t) * (isWaterPlant ? expf(-0.1f*terrain.w) : 1.f);
			const float lightIntervalMax = 0.5f * (__half2float((*c_parameters.vegTypeBuffer).lightConditions[veg.type].x) + __half2float((*c_parameters.vegTypeBuffer).lightConditions[veg.type].y));
			const float shadowGrowth = 1.f - 2.f * abs(shadowVal - lightIntervalMax) / (__half2float((*c_parameters.vegTypeBuffer).lightConditions[veg.type].y) - __half2float((*c_parameters.vegTypeBuffer).lightConditions[veg.type].x));

			// Max radius based on neighborhood and terrain
                        const float baseMaxRadius =
                                fminf(fmaxf(__half2float(veg.pos_r.b.x) - bedrockHeight, 0.f) /
                                              __half2float((*c_parameters.vegTypeBuffer).height[veg.type].y),
                                      fmaxf(1.f - overlap, 0.f) * fmaxf(shadowGrowth, 0.f) *
                                              __half2float((*c_parameters.vegTypeBuffer).maxRadius[veg.type]));
                        const float waterMaxRadius = fminf(
                                baseMaxRadius,
                                fmaxf((waterLevel - __half2float(veg.pos_r.b.x)) /
                                              __half2float((*c_parameters.vegTypeBuffer).height[veg.type].x),
                                      0.f));
			const float maxRadius = isWaterPlant ? waterMaxRadius : baseMaxRadius;

			// 1.1 * maxRadius serves as a Buffer and prevents oscillations
			// Shrink, if possible
                        veg.pos_r.b.y -= __float2half(clamp(
                                __half2float((*c_parameters.vegTypeBuffer).shrinkRate[veg.type]) *
                                        c_parameters.deltaTime *
                                        (__half2float(veg.pos_r.b.y) > 1.1f * maxRadius ? 1.f : 0.f),
                                0.f,
                                __half2float(veg.pos_r.b.y) - maxRadius));

			// Damage plant if it is still too large given the current conditions
                        const float radiusDamage = fmaxf((__half2float(veg.pos_r.b.y) - 1.1f * maxRadius) /
                                              __half2float(veg.pos_r.b.y),
                                      0.f);
			const float radiusRate = radiusDamage > 0.f ? 0.f : 1.f;

			// Calculate growth and health
			const float growthRate = fmaxf(shadowGrowth, 0.f) * radiusRate * slopeGrowth * moistureGrowth * thirstGrowth * soilRate * waterGrowth * fmaxf(1.f - overlap, 0.f) * c_parameters.deltaTime * __half2float((*c_parameters.vegTypeBuffer).growthRate[veg.type]);
			veg.health -= __half2float((*c_parameters.vegTypeBuffer).damageRate[veg.type]) * c_parameters.deltaTime * (fmaxf(-shadowGrowth, 0.f) + fmaxf(waterDamage, 0.f) + fmaxf(rootDamage, 0.f) + fmaxf(stemDamage, 0.f) + thirstDamage + moistureDamage + slopeDamage + radiusDamage);
			// TODO: maybe also a constant rate?
			veg.health += growthRate;

			// Calculate maximum radius based on root depth and bedrock and grow plant
                        const float newRadius =
                                fminf(__half2float(veg.pos_r.b.y) + growthRate, maxRadius);
                        veg.pos_r.b.y = __float2half(newRadius > __half2float(veg.pos_r.b.y)
                                             ? newRadius
                                             : __half2float(veg.pos_r.b.y));

			// Check if maturity condition met
                        if(veg.age > (*c_parameters.vegTypeBuffer).maxMaturityTime[veg.type] &&
                           veg.pos_r.b.y <
                                   (*c_parameters.vegTypeBuffer).maturityPercentage[veg.type] *
                                           (*c_parameters.vegTypeBuffer).maxRadius[veg.type])
                        {
				veg.health = CUDART_ZERO_FP16;
			}
			veg.health = __hmax(__hmin(veg.health, CUDART_ONE_FP16), CUDART_ZERO_FP16);
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
		const float4 terrain = half4toFloat4(c_parameters.terrainArray.read(vegCell));
        const float  zpos    = terrain.x + terrain.y + terrain.z;
        const float  maxRadius =
                fminf(fmaxf(zpos - terrain.x, 0.f) /
                                __half2float((*c_parameters.vegTypeBuffer).height[veg.type].y),
                        __half2float((*c_parameters.vegTypeBuffer).maxRadius[veg.type]));

                veg.pos_r            = {
                        __float22half2_rn((make_float2 (vegCell) + 0.5f) * c_parameters.gridScale),
                __float22half2_rn(float2 {
                        zpos,
                        0.05f * maxRadius + 0.95f * maxRadius * random::uniform_float(seed.z)})};
		c_parameters.vegBuffer[idx] = veg;
	}

	__global__ void finishSort(int count)
	{
		const int idx = getGlobalIndex1D();

		if (idx >= count) {
			return;
		}
		c_parameters.vegBuffer[idx] = c_parameters.adaptiveGrid.vegBuffer[c_parameters.adaptiveGrid.indexBuffer[idx]];
	}

	__global__ void fillAdaptiveKeys(int count) 
	{
		const int idx = getGlobalIndex1D();

		if (idx >= count) {
			return;
		}

		const Vegetation veg = c_parameters.vegBuffer[idx];
		c_parameters.adaptiveGrid.vegBuffer[idx] = veg;
		c_parameters.adaptiveGrid.indexBuffer[idx] = idx;

		float2 pos           = __half22float2(veg.pos_r.a);
        float  radius        = __half2float(veg.pos_r.b.y);

		constexpr unsigned int maxIndex = (unsigned int)-1;

		if (veg.health < CUDART_ZERO_FP16) 
		{
			c_parameters.adaptiveGrid.keyBuffer[idx] = maxIndex;
			atomicAdd(c_parameters.vegCountBuffer, -1);

			return;
		}

		for (unsigned int i = 0; i < c_parameters.adaptiveGrid.layerCount; ++i)
		{
			if ((radius <= 0.5f * c_parameters.adaptiveGrid.gridScales[i]) || (i == c_parameters.adaptiveGrid.layerCount - 1))
			{
				const int2 adaptiveCell = getWrappedCell(make_int2(pos / c_parameters.adaptiveGrid.gridScales[i]), c_parameters.adaptiveGrid.gridSizes[i]); // Wrapped Cell?
				c_parameters.adaptiveGrid.keyBuffer[idx] = (i << (32u - c_parameters.adaptiveGrid.layerBits)) | (unsigned int)getCellIndex(adaptiveCell, c_parameters.adaptiveGrid.gridSizes[i]);

				return;
			}
		}
	}
	
	__global__ void findGridStart(int count) {
		const int idx = getGlobalIndex1D();
		if (idx >= count - 1) {
			return;
		}
		// Adaptive Grid
		const unsigned int adaptiveKeyA = c_parameters.adaptiveGrid.keyBuffer[idx];
		const unsigned int adaptiveKeyB = c_parameters.adaptiveGrid.keyBuffer[idx + 1];
		const unsigned int adaptiveLayerA = (c_parameters.adaptiveGrid.layerMask & adaptiveKeyA) >> (32u - c_parameters.adaptiveGrid.layerBits);
		const unsigned int adaptiveIndexA = c_parameters.adaptiveGrid.indexMask & adaptiveKeyA;
		const unsigned int adaptiveLayerB = (c_parameters.adaptiveGrid.layerMask & adaptiveKeyB) >> (32u - c_parameters.adaptiveGrid.layerBits);
		const unsigned int adaptiveIndexB = c_parameters.adaptiveGrid.indexMask & adaptiveKeyB;

		if (adaptiveKeyA != adaptiveKeyB) {

			c_parameters.adaptiveGrid.gridBuffer[adaptiveLayerA][adaptiveIndexA + 1] = idx + 1;
			c_parameters.adaptiveGrid.gridBuffer[adaptiveLayerB][adaptiveIndexB] = idx + 1;
		}

		if (idx == 0) {
			c_parameters.adaptiveGrid.gridBuffer[adaptiveLayerA][adaptiveIndexA] = 0;
		}
		if (idx == (count - 2)) {
			c_parameters.adaptiveGrid.gridBuffer[adaptiveLayerB][adaptiveIndexB + 1] = count;
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

		const float2 heights{ __half22float2(c_parameters.vegHeightArray.read(cell)) }; // .x is height without Veg, .y is height with Veg
		float2 shadow{ 0.f, 0.f }; // .x is shadow at ground level, .y is shadow at top of vegetation
		const float2 lightDirection = { -sqrtf(2.f), -sqrtf(2.f) };

		const float2 position{ make_float2(cell) + 0.5f };

		for (int i = -3; i <= 3; ++i) {
			for (int j = -3; j <= 3; ++j) {
				const float2 offset = make_float2(int2{ i, j });
				const float distance = length(offset) * c_parameters.gridScale;
				float2 nextPosition = position - offset;

				const float nextHeight{ sampleLinearOrNearest<true>(c_parameters.vegHeightArray, nextPosition).y }; // .y is the height including Vegetation
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

		c_parameters.shadowArray.write(cell, __float22half2_rn(clamp(2.f * (shadow - 0.5f), 0.f, 1.f))); // Rescaling shadow, because the dot product means that shadow can't be smaller 0.5 than 0.5 (roughly)
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

	void initAdaptiveGrid(const SimulationParameters& t_simulationParameters)
	{
		size_t cellCount = 0;

		for (int i = 0; i < t_simulationParameters.adaptiveGrid.layerCount; ++i)
		{
			cellCount += t_simulationParameters.adaptiveGrid.cellCounts[i];
		}

		CU_CHECK_ERROR(cudaMemset(t_simulationParameters.adaptiveGrid.gridBuffer[0], 0xFFFFFFFF, cellCount * sizeof(unsigned int)));
	}

	void vegetation(LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters, std::vector<sthe::cu::Stopwatch>& watches) {
		watches[2].start();
		getVegetationCount(t_launchParameters, t_simulationParameters);
		int count = t_launchParameters.vegCount;

		initAdaptiveGrid(t_simulationParameters);
		
		Buffer<half> slopeBuffer{ t_launchParameters.tmpBuffer + t_simulationParameters.cellCount };
		Buffer<int> relMapBuffer{ reinterpret_cast<Buffer<int>>(slopeBuffer + 2 * t_simulationParameters.cellCount) };

		if (count > 0) {
			fillAdaptiveKeys<<<t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (count);

			thrust::sort_by_key(thrust::device, t_simulationParameters.adaptiveGrid.keyBuffer, t_simulationParameters.adaptiveGrid.keyBuffer + count, t_simulationParameters.adaptiveGrid.indexBuffer);

			finishSort<<<t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D>>>(count);

			// memset type counter 0
			CU_CHECK_ERROR(cudaMemset(t_simulationParameters.vegCountBuffer + 1, 0, c_maxVegTypeCount * sizeof(int)));

			// atomic adds for type counter
			// return of atomic in tmp buffer for veg id
			prepareVegMapKernel<<<t_launchParameters.optimalGridSize1D, t_launchParameters.optimalBlockSize1D>>>(relMapBuffer);
			
			// second pass use relative offset in tmp buffer + global counter offset to write global id in map 
			vegMapKernel<<<t_launchParameters.optimalGridSize1D, t_launchParameters.optimalBlockSize1D>>>(relMapBuffer);

			getVegetationCount(t_launchParameters, t_simulationParameters);
			count = t_launchParameters.vegCount;

			findGridStart<<<t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D>>>(count);
		}
		watches[2].stop();

		watches[3].start();
		growVegetation<<<t_launchParameters.vegetationGridSize1D, t_launchParameters.blockSize1D >> > (count, slopeBuffer);
		watches[3].stop();
		watches[4].start();
		rasterizeVegetation<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (slopeBuffer, count);
		watches[4].stop();
		watches[5].start();
		calculateShadowMap << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > ();
		watches[5].stop();
	}
}
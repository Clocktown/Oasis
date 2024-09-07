#pragma once

#include <sthe/device/buffer.cuh>
#include <sthe/device/array2d.cuh>
#include <cuda_runtime.h>
#include <cufft.h>

#define TAN10 0.1763f
#define TAN15 0.2679f
#define TAN33 0.6494f
#define TAN45 1.0f
#define TAN55 1.4281f
#define TAN68 2.5f

namespace dunes
{

template<typename T>
using Array2D = sthe::device::Array2D<T>;

template<typename T>
using Buffer = sthe::device::Buffer<T>;

struct WindWarping
{
	int count{ 2 };
	float i_divisor{ 1.f / 20.0f };
	float radii[4]{ 200.0f, 50.0f, 0.0f, 0.0f };
	float strengths[4]{ 0.8f, 0.2f, 0.0f, 0.0f };
	float gradientStrengths[4]{ 30.f, 5.f, 0.0f, 0.0f };
	Buffer<cuComplex> gaussKernels[4];
	Buffer<cuComplex> smoothedHeights[4];
};

struct VegetationType
{
	float maxRadius; // Plant is mature at maturityPercentage of maxRadius
	float growthRate;
	float positionAdjustRate; // how fast the plant can change its position (height) to match the terrain
	float damageRate; // How resistant the plant is to being damaged, lower values cause it to be able to survive for longer in bad environments
	float shrinkRate; // if > 0.f, plant is able to "shrink" to a smaller size if environment doesn't support its current size anymore
	float maxMaturityTime; // If Plant hasn't reached maturity after this time, it dies
	float maturityPercentage; // %radius needed to be mature
	float2 height; // .x * maxRadius = height above ground; .y * maxRadius = root depth; relevant for vegetation density and growth
	float waterUsageRate;
	float waterStorageCapacity;
	float waterResistance; // How well the plant can handle standing water. >= 1 is a water plant, which follows different rules
	float maxMoisture; // Plant takes damage when moisture is outside this interval, more damage the more it is outside, gains health inside interval, more health towards middle of interval
	float soilCompatibility; // controls growth in soil
	float sandCompatibility; // controls growth in sand
	float2 terrainCoverageResistance; // .x threshold for how much roots need to be covered; .y threshold for how much of stem is allowed to be covered
	float maxSlope;
	float baseSpawnRate;
	float densitySpawnMultiplier;
	float windSpawnMultiplier;
	float humusRate;
};

struct Vegetation // needs to be aligned for gl
{
	float3 pos{}; // pos where root and stem start
	int type{ 0 };
	float health{ 1.f };
	float water{ 0.f };
	float age{ 0.f };
	float radius{ 0.f };
};

struct SimulationParameters
{
	int2 gridSize{ 2048, 2048 };
	int2 uniformGridSize{ 204, 204 };
	float uniformGridScale{ 2048.f / 204.f };
	float rUniformGridScale{ 1.f / (2048.f / 204.f) };
	int uniformGridCount{ uniformGridSize.x * uniformGridSize.y };
	float gridScale{ 1.0f };
	float rGridScale{ 1.0f / gridScale };
	int cellCount{ gridSize.x * gridSize.y };

	float2 windDirection{ 1.0f, 0.0f };
	float windSpeed{ 10.0f };

	float venturiStrength{ 0.005f };

	float windShadowDistance{ 1.0f };
	float minWindShadowAngle{ TAN10 };
	float maxWindShadowAngle{ TAN15 };

	float abrasionStrength{ 0.0f };
	float soilAbrasionStrength{ 0.0f };
	float abrasionThreshold{ 0.1f };
	float saltationStrength{ 0.05f };
	float reptationStrength{ 0.0f };
	float reptationSmoothingStrength{ 0.0f };
	float reptationUseWindShadow{ 0.f };

	float avalancheAngle{ TAN33 };
	float bedrockAngle{ TAN68 };
	float vegetationAngle{ TAN45 };
	float soilAngle{ TAN45 };
	float vegetationSoilAngle{ TAN68 };

	float wavePeriod{ 0.02f };
	float waveStrength{ 0.005f };
	float waveDepthScale{ 0.1f };

	float sedimentCapacityConstant{ 0.1f };
	float sedimentDepositionConstant{ 0.1f };
	float sandDissolutionConstant{ 0.1f };
	float soilDissolutionConstant{ 0.05f };
	float bedrockDissolutionConstant{ 0.01f };

	float waterBorderLevel{ 20.f };
	float waterLevel{ 0.f };

	float moistureEvaporationScale{ 1.f };
	float sandMoistureRate{ 0.1f };
	float soilMoistureRate{ 0.02f };
	float iTerrainThicknessMoistureThreshold{ 1.f };
	float moistureCapacityConstant{ 1.f };

	float evaporationRate{ 0.01f };
	float rainStrength{10.f};
	float rainPeriod{0.2f};
	float rainScale{30.f};
	float rainProbabilityMin{0.5f};
	float rainProbabilityMax{1.f};
	float iRainProbabilityHeightRange{0.001f};

	int maxVegCount{ 100000 };
	int vegTypeCount{ 3 };

	float deltaTime{ 1.0f };
	int timestep = 0;

	Array2D<float4> terrainArray;
	Array2D<float2> windArray;
	Array2D<float4> resistanceArray; // .x = wind shadow, .y = vegetation, .z = erosion, .w = humus
	Array2D<float4> fluxArray;
	Array2D<float2> velocityArray;
	Array2D<float>	sedimentArray;
	Array2D<float> moistureArray;
	Array2D<float2> shadowArray;
	Array2D<float2> vegHeightArray;
	Buffer<Vegetation> vegBuffer;
	Buffer<int> vegMapBuffer;
	Buffer<uint4> seedBuffer;
	Buffer<int> vegCountBuffer;
	Buffer<VegetationType> vegTypeBuffer;
	Buffer<float> vegMatrixBuffer;

	Buffer<uint2> uniformGrid;
	Buffer<unsigned int> keyBuffer; // 1 * max vegetation count
	//Buffer<unsigned int> indexBuffer; // 1 * max vegetation count

	Buffer<cuComplex> shadowMapKernel;
	Buffer<cuComplex> shadowMapSmoothed;
};

void upload(const SimulationParameters& t_simulationParameters);

}

#undef TAN10
#undef TAN15
#undef TAN33
#undef TAN45
#undef TAN55
#undef TAN68

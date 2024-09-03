#include <dunes/core/simulation_parameters.hpp>
#include <cuda_runtime.h>
#include "constants.cuh"

#define SQRT2 1.414213562f
#define RSQRT2 0.707106782f

namespace dunes
{

__constant__ SimulationParameters c_parameters{};

__constant__ int2 c_offsets[8]{ int2{ 1, 0 }, int2{ 1, 1 }, int2{ 0, 1 }, int2{ -1, 1 },
                                int2{ -1, 0 }, int2{ -1, -1 }, int2{ 0, -1 }, int2{ 1, -1 } };

__constant__ float c_distances[8]{ 1.0f, SQRT2, 1.0f, SQRT2, 1.0f, SQRT2, 1.0f, SQRT2 };
__constant__ float c_rDistances[8]{ 1.0f, RSQRT2, 1.0f, RSQRT2, 1.0f, RSQRT2, 1.0f, RSQRT2 };
// TODO: Dominance/compatibility Matrix
__constant__ float c_vegetationMatrix[c_numVegetationTypes][c_numVegetationTypes]{
    {1.f, 1.f, 1.f},
    {1.f, 1.f, 1.f},
    {1.f, 1.f, 1.f}
};

//struct VegetationType
//{
//	float maxRadius; // Plant is mature at maturityPercentage of maxRadius
//	float growthRate;
//	float maxMaturityTime; // If Plant hasn't reached maturity after this time, it dies
//	float maturityPercentage; // %radius needed to be mature
//	float2 height; // .x * maxRadius = height above ground; .y * maxRadius = root depth; relevant for vegetation density and growth
//	float waterUsageRate;
//	float waterStorageCapacity;
//	float waterResistance; // How well the plant can handle standing water. >= 1 is a water plant, which follows different rules
//	float maxMoisture; // Plant takes damage when moisture is outside this interval, more damage the more it is outside, gains health inside interval, more health towards middle of interval
//	float soilCompatibility; // controls growth in soil
//	float sandCompatibility; // controls growth in sand
//	float2 terrainCoverageResistance; // .x threshold for how much roots need to be covered; .y threshold for how much of stem is allowed to be covered
//	float maxSlope;
//	float baseSpawnRate;
//	float densitySpawnMultiplier;
//	float windSpawnMultiplier;
//	float humusRate;
//};

__constant__ VegetationType c_vegTypes[c_numVegetationTypes]{
    {
        // Trees or something
        20.f,
        0.1f,
        100.f,
        0.2f,
        {2.f, 1.f},
        0.1f,
        1.f,
        0.1f,
        0.8f,
        1.f,
        0.2f,
        {0.75f, 0.25f},
        0.5f,
        0.001f,
        10.f,
        30.f,
        0.01f
    }, 
        {
        // Grass (for touching)
        2.f,
        0.2f,
        20.f,
        0.2f,
        {2.f, 0.5f},
        0.05f,
        2.f,
        0.3f,
        0.9f,
        1.f,
        0.4f,
        {0.75f, 0.75f},
        0.75f,
        0.001f,
        10000000.f,
        10.f,
        0.01f
    }, 
    {
        // Seaweed or something
        1.f,
        0.3f,
        30.f,
        0.1f,
        {3.f, 1.f},
        0.1f,
        1.f,
        1.0f,
        1.f,
        1.f,
        0.2f,
        {0.75f, 0.25f},
        0.25f,
        0.01f,
        1.f,
        0.1f,
        0.001f
    } 
};
}

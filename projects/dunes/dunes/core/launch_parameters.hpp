#pragma once

#include "simulation_parameters.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <array>
#include <vector>

namespace dunes
{

enum class TimeMode : unsigned char
{
	DeltaTime, FixedDeltaTime
};

enum class SaltationMode : unsigned char
{
	Backward, Forward
};

enum class WindWarpingMode : unsigned char
{
	None, Standard
};

enum class WindShadowMode : unsigned char
{
	Linear, Curved
};

enum class BedrockAvalancheMode : unsigned char
{
	ToSand, ToBedrock
};

enum class ProjectionMode : unsigned char
{
	None, Jacobi, FFT
};

struct Projection
{
	ProjectionMode mode{ ProjectionMode::Jacobi };
	int jacobiIterations{ 50 };
	cufftHandle planR2C;
	cufftHandle planC2R;
	Buffer<float> velocities[2];
};

struct Vegetation
{
	int type{ 0 };
	float3 pos{}; // pos where root and stem start
	float2 height{}; // .x height starting at pos upwards; .y height starting at pos downwards (roots)
	float radius{ 0.f };
};

struct LaunchParameters
{
	unsigned int blockSize1D;
	dim3 blockSize2D;
	unsigned int gridSize1D;
	dim3 gridSize2D;

	unsigned int uniformGridSize1D;
	unsigned int vegetationGridSize1D;

	unsigned int optimalBlockSize1D;
	dim3 optimalBlockSize2D;
	unsigned int optimalGridSize1D;
	dim3 optimalGridSize2D;

	SaltationMode saltationMode{ SaltationMode::Forward };
	WindWarpingMode windWarpingMode{ WindWarpingMode::None };
	WindShadowMode windShadowMode{ WindShadowMode::Linear };
	BedrockAvalancheMode bedrockAvalancheMode{ BedrockAvalancheMode::ToSand };
	bool useBilinear{ true };
	int avalancheIterations{ 50 };
	int bedrockAvalancheIterations{ 2 };
	int soilAvalancheIterations{ 2 };
	TimeMode timeMode{ TimeMode::DeltaTime };

	Array2D<float4> terrainArray;
	Array2D<float2> windArray;
	Array2D<float4> resistanceArray; // .x = wind shadow, .y = vegetation, .z = erosion, .w = sticky
	Buffer<float> slabBuffer;
	Buffer<float> tmpBuffer; // 4 * gridSize.x * gridSize.y
	Buffer<Vegetation> vegBuffer;
	Buffer<uint4> seedBuffer;
	Buffer<int> vegetationCount;
	int numVegetation{ 100 };
	int maxVegetation{ 100000 };
	Buffer<uint2> uniformGrid;
	Buffer<unsigned int> keyBuffer; // 1 * max vegetation count
	//Buffer<unsigned int> indexBuffer; // 1 * max vegetation count
	WindWarping windWarping;
	Projection projection;

	cufftHandle fftPlan{ 0 };
};

struct NoiseGenerationParameters 
{
	float2 offset{ 0.f, 0.f };
	float2 stretch{ 1.f, 1.f };
	float2 border{ 0.1f , 0.1f };
    float scale = 100.f;
    float bias = 0.f;
    int iters = 0;
	bool flat = false;
	bool enabled = true;
	bool uniform_random = false;
};

constexpr int NumNoiseGenerationTargets = 6;

enum class NoiseGenerationTarget : unsigned char
{
	Bedrock, Sand, Vegetation, AbrasionResistance, Soil, Water
};

struct InitializationParameters
{
	NoiseGenerationParameters noiseGenerationParameters[NumNoiseGenerationTargets]{
		{},
		{{ 0.f, 0.f }, { 1.f, 1.f }, { 0.1f , 0.1f }, 100.f, 1.f, 0, true, true, false},
		{{ 0.f, 0.f }, { 1.f, 1.f }, { 0.1f , 0.1f }, 1.f, 0.f, 0, true, true, false},
		{{ 0.f, 0.f }, { 1.f, 1.f }, { 0.1f , 0.1f }, 1.f, 0.f, 0, true, true, false},
		{{ 0.f, 0.f }, { 1.f, 1.f }, { 0.1f , 0.1f }, 100.f, 10.f, 0, true, true, false},
		{{ 0.f, 0.f }, { 1.f, 1.f }, { 0.1f , 0.1f }, 100.f, 5.f, 0, true, true, false}
	};
};

}

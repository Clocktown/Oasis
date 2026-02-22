#pragma once

#include "simulation_parameters.hpp"
#include <dunes/device/constants.cuh>
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

struct LaunchParameters
{
	unsigned int blockSize1D;
	dim3 blockSize2D;
	unsigned int gridSize1D;
	dim3 gridSize2D;

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

	int vegCount{ 0 };
	int vegCountsPerType[c_maxVegTypeCount]{};
	int maxVegCount{ 1000000 };

	WindWarping windWarping;
	Projection projection;
	Buffer<float> tmpBuffer; // 5 * gridSize.x * gridSize.y

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

constexpr int NumNoiseGenerationTargets = 7;

enum class NoiseGenerationTarget : unsigned char
{
	Bedrock, Sand, Vegetation, AbrasionResistance, Soil, Water, Moisture
};

struct InitializationParameters
{
	NoiseGenerationParameters noiseGenerationParameters[NumNoiseGenerationTargets]{
		{},
		{{ 0.f, 0.f }, { 1.f, 1.f }, { 0.1f , 0.1f }, 100.f, 1.f, 0, true, true, false},
		{{ 0.f, 0.f }, { 1.f, 1.f }, { 0.1f , 0.1f }, 1.f, 0.f, 0, true, true, false},
		{{ 0.f, 0.f }, { 1.f, 1.f }, { 0.1f , 0.1f }, 1.f, 0.f, 0, true, true, false},
		{{ 0.f, 0.f }, { 1.f, 1.f }, { 0.1f , 0.1f }, 100.f, 10.f, 0, true, true, false},
		{{ 0.f, 0.f }, { 1.f, 1.f }, { 0.1f , 0.1f }, 100.f, 20.f, 0, true, true, false},
		{{ 0.f, 0.f }, { 1.f, 1.f }, { 0.1f , 0.1f }, 100.f, 0.3f, 0, true, true, false}
	};
};

}

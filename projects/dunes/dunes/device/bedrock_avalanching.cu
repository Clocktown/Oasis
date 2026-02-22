#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void setupBedrockAvalancheKernel(Buffer<half4> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	t_terrainBuffer[cellIndex] = c_parameters.terrainArray.read(cell);
}

template<BedrockAvalancheMode mode>
__global__ void bedrockAvalancheKernel(Buffer<half4> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	const float4 terrain{ half4toFloat4(t_terrainBuffer[cellIndex]) };
	const float height{ terrain.x };
	
	int nextCellIndices[8];
	float avalanches[8];
	float avalancheSum{ 0.0f };
	float maxAvalanche{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		nextCellIndices[i] = getCellIndex(getWrappedCell(cell + c_offsets[i]));
        const float4 nextTerrain{ half4toFloat4(t_terrainBuffer[nextCellIndices[i]]) };
		const float nextHeight{ nextTerrain.x };

		const float heightDifference{ height - nextHeight };
		avalanches[i] = fmaxf(heightDifference - c_parameters.bedrockAngle * c_distances[i] * c_parameters.gridScale, 0.0f);
		avalancheSum += avalanches[i];
		maxAvalanche = fmaxf(maxAvalanche, avalanches[i]);
	}

	if (avalancheSum > 0.0f)
	{
		const float rAvalancheSum{ 1.0f / avalancheSum };
		const float avalancheSize{ maxAvalanche / (1.0f + maxAvalanche * rAvalancheSum) };


		const float scale{ avalancheSize * rAvalancheSum };

		for (int i{ 0 }; i < 8; ++i)
		{
			if (avalanches[i] > 0.0f)
			{
				if constexpr (mode == BedrockAvalancheMode::ToSand)
				{
					atomicAdd(&t_terrainBuffer[nextCellIndices[i]].a.y, __float2half(scale * avalanches[i]));
				}
				else
				{
					atomicAdd(&t_terrainBuffer[nextCellIndices[i]].a.x, __float2half(scale * avalanches[i]));
				}
			}
		}

		atomicAdd(&t_terrainBuffer[cellIndex].a.x, __float2half(- avalancheSize));
	}
}

__global__ void soilAvalancheKernel(Buffer<half4> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	const float4 terrain{ half4toFloat4(t_terrainBuffer[cellIndex]) };
	const float height{ terrain.x + terrain.z };

	const float vegetation{ __half2float(c_parameters.resistanceArray.read(cell).a.y) };

	const float terrainThickness = terrain.y + terrain.z;
	const float moistureCapacityConstant = c_parameters.moistureCapacityConstant;
	const float moistureCapacity = moistureCapacityConstant * clamp(terrainThickness * c_parameters.iTerrainThicknessMoistureThreshold, 0.f, 1.f);
    const float moisture{ 2.f * clamp(__half2float(c_parameters.moistureArray.read(cell)) / (moistureCapacity + 1e-6f), 0.f, 1.f) - 1.f  };

	const float moistureFactor = moisture > 0.f ?
		1.5f - 1.4f * moisture :
		1.5f + 0.5f * moisture;

	const float moistureVegetationFactor = moisture > 0.f ?
		1.5f - 1.3f * moisture :
		1.5f + 0.5f * moisture;

	// Store precomputed angle
	const float soilAngle = lerp(moistureFactor * c_parameters.soilAngle, moistureVegetationFactor * c_parameters.vegetationSoilAngle, fmaxf(vegetation, 0.f));
	
	int nextCellIndices[8];
	float avalanches[8];
	float avalancheSum{ 0.0f };
	float maxAvalanche{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		nextCellIndices[i] = getCellIndex(getWrappedCell(cell + c_offsets[i]));
        const float4 nextTerrain{ half4toFloat4(t_terrainBuffer[nextCellIndices[i]]) };
		const float nextHeight{ nextTerrain.x + nextTerrain.z };

		const float heightDifference{ height - nextHeight };
		avalanches[i] = fmaxf(heightDifference - soilAngle * c_distances[i] * c_parameters.gridScale, 0.0f);
		avalancheSum += avalanches[i];
		maxAvalanche = fmaxf(maxAvalanche, avalanches[i]);
	}

	if (avalancheSum > 0.0f)
	{
		const float rAvalancheSum{ 1.0f / avalancheSum };
		const float avalancheSize{  fminf(maxAvalanche / (1.0f + maxAvalanche * rAvalancheSum), terrain.z) };


		const float scale{ avalancheSize * rAvalancheSum };

		for (int i{ 0 }; i < 8; ++i)
		{
			if (avalanches[i] > 0.0f)
			{
				atomicAdd(&t_terrainBuffer[nextCellIndices[i]].b.x, __float2half(scale * avalanches[i]));
			}
		}

		atomicAdd(&t_terrainBuffer[cellIndex].b.x, __float2half(- avalancheSize));
	}
}

__global__ void finishBedrockAvalancheKernel(Buffer<half4> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	c_parameters.terrainArray.write(cell, t_terrainBuffer[cellIndex]);
}

void bedrockAvalanching(const LaunchParameters& t_launchParameters)
{
	if (t_launchParameters.bedrockAvalancheIterations <= 0 && t_launchParameters.soilAvalancheIterations <= 0)
	{
		return;
	}

	Buffer<half4> terrainBuffer{ reinterpret_cast<Buffer<half4>>(t_launchParameters.tmpBuffer) };
	setupBedrockAvalancheKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(terrainBuffer);

	if (t_launchParameters.bedrockAvalancheMode == BedrockAvalancheMode::ToSand)
	{
		for (int i = 0; i < t_launchParameters.bedrockAvalancheIterations; ++i)
		{
			bedrockAvalancheKernel<BedrockAvalancheMode::ToSand><<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(terrainBuffer);
		}
	}
	else
	{
		for (int i = 0; i < t_launchParameters.bedrockAvalancheIterations; ++i)
		{
			bedrockAvalancheKernel<BedrockAvalancheMode::ToBedrock><<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(terrainBuffer);
		}
	}

	for (int i = 0; i < t_launchParameters.soilAvalancheIterations; ++i)
	{
		soilAvalancheKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(terrainBuffer);
	}

	finishBedrockAvalancheKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(terrainBuffer);
}

}

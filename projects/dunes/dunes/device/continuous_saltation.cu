#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>
#include <sthe/config/debug.hpp>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <cstdio>

namespace dunes
{

	__global__ void setupContinuousSaltationKernel(Buffer<half> t_advectedSlabBuffer)
	{
		const int2 index{ getGlobalIndex2D() };
		const int2 stride{ getGridStride2D() };

		int2 cell;

		for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
		{
			for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
			{
				float4 terrain{ half4toFloat4(c_parameters.terrainArray.read(cell)) };

				const float2 windVelocity {sampleLinearOrNearest<true>(
                                        c_parameters.windArray, 0.5f * (make_float2(cell) + 0.5f))};
				const float windSpeed{ length(windVelocity) };

				const float terrainThickness = terrain.y + terrain.z;
				const float moistureCapacityConstant = c_parameters.moistureCapacityConstant;
				const float moistureCapacity = moistureCapacityConstant * clamp(terrainThickness * c_parameters.iTerrainThicknessMoistureThreshold, 0.f, 1.f);
				const float moisture{ clamp(__half2float(c_parameters.moistureArray.read(cell)) / (moistureCapacity + 1e-6f), 0.f, 1.f) };
				const float moistureFactor{ clamp(1.f - 10.f * moisture, 0.f, 1.f) };

				const float4 resistance{ half4toFloat4(c_parameters.resistanceArray.read(cell)) };
				const float saltationScale{ (terrain.w > 1e-5f ? 0.f : 1.f) * (1.0f - resistance.x) * (1.0f - fmaxf(resistance.y, 0.f)) * moistureFactor };

				// TODO: lower saltation when cell is wet
				//const float scale{ windSpeed * c_parameters.deltaTime };

				const float saltation{ fminf(c_parameters.saltationStrength * saltationScale, terrain.y) };

				terrain.y -= saltation;
				c_parameters.terrainArray.write(cell, toHalf4(terrain));

				const int cellIndex{ getCellIndex(cell) };
				const float slab{ saltation };

				c_parameters.slabBuffer[cellIndex] += __float2half(slab);
				t_advectedSlabBuffer[cellIndex] = CUDART_ZERO_FP16;
			}
		}
	}

	template <bool TUseBilinear>
	__global__ void continuousSaltationKernel(Buffer<half> t_advectedSlabBuffer)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int cellIndex{ getCellIndex(cell) };
		const float slab{ c_parameters.slabBuffer[cellIndex] };

		const float2 windVelocity {sampleLinearOrNearest<true>(
                        c_parameters.windArray, 0.5f * (make_float2(cell) + 0.5f))};

		const float2 position{ make_float2(cell) };

		if (slab > 0.0f)
		{
			const float2 nextPosition{ position + windVelocity * c_parameters.rGridScale * c_parameters.deltaTime };

			if constexpr (TUseBilinear) {
				const int2 nextCell{ make_int2(floorf(nextPosition)) };

				for (int x{ nextCell.x }; x <= nextCell.x + 1; ++x)
				{
					const float u{ 1.0f - abs(static_cast<float>(x) - nextPosition.x) };

					for (int y{ nextCell.y }; y <= nextCell.y + 1; ++y)
					{
						const float v{ 1.0f - abs(static_cast<float>(y) - nextPosition.y) };
						const float weight{ u * v };

						if (weight > 0.0f)
						{
							atomicAdd(t_advectedSlabBuffer + getCellIndex(getWrappedCell(int2{ x,y })), __float2half(weight * slab));
						}
					}
				}
			}
			else {
				const int2 nextCell{ getNearestCell(nextPosition) };
				atomicAdd(t_advectedSlabBuffer + getCellIndex(getWrappedCell(nextCell)), __float2half(slab));
			}
		}
	}

	template <bool TUseBilinear>
	__global__ void continuousBackwardSaltationKernel(Buffer<half> t_advectedSlabBuffer)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int cellIndex{ getCellIndex(cell) };
		float slab{ 0.f };

		const float2 windVelocity {sampleLinearOrNearest<true>(
                        c_parameters.windArray, 0.5f * (make_float2(cell) + 0.5f))};

		const float2 position{ make_float2(cell) };

		const float2 nextPosition{ position - windVelocity * c_parameters.rGridScale * c_parameters.deltaTime };

		if constexpr (TUseBilinear) {
			const int2 nextCell{ make_int2(floorf(nextPosition)) };

			for (int x{ nextCell.x }; x <= nextCell.x + 1; ++x)
			{
				const float u{ 1.0f - abs(static_cast<float>(x) - nextPosition.x) };

				for (int y{ nextCell.y }; y <= nextCell.y + 1; ++y)
				{
					const float v{ 1.0f - abs(static_cast<float>(y) - nextPosition.y) };
					const float weight{ u * v };

					if (weight > 0.0f)
					{
						slab += __half2float(c_parameters.slabBuffer[getCellIndex(getWrappedCell(int2{ x,y }))]) * weight;
					}
				}
			}
		}
		else {
			const int2 nextCell{ getNearestCell(nextPosition) };
			slab += __half2float(c_parameters.slabBuffer[getCellIndex(getWrappedCell(nextCell))]);
		}


		t_advectedSlabBuffer[cellIndex] = __float2half(slab);
	}

	__global__ void finishContinuousSaltationKernel(Buffer<half> t_advectedSlabBuffer)
	{
		const int2 index{ getGlobalIndex2D() };
		const int2 stride{ getGridStride2D() };

		int2 cell;

		for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
		{
			for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
			{
				const int cellIndex{ getCellIndex(cell) };

				float4 terrain{ half4toFloat4(c_parameters.terrainArray.read(cell)) };

				const float terrainThickness = terrain.y + terrain.z;
				const float moistureCapacityConstant = c_parameters.moistureCapacityConstant;
				const float moistureCapacity = moistureCapacityConstant * clamp(terrainThickness * c_parameters.iTerrainThicknessMoistureThreshold, 0.f, 1.f);
				const float moisture{ clamp(__half2float(c_parameters.moistureArray.read(cell)) / (moistureCapacity + 1e-6f), 0.f, 1.f) };
				const float abrasionMoistureFactor{ clamp(1.f - 2.f * moisture, 0.f, 1.f) };
				const float saltationMoistureFactor{ clamp(1.f - 10.f * moisture, 0.01f, 1.f) };

				const float slab{ t_advectedSlabBuffer[cellIndex] };

				const float windSpeed {length(sampleLinearOrNearest<true>(
                                        c_parameters.windArray,
                                        0.5f * (make_float2(cell) + 0.5f)))};

				const float4 resistance{ half4toFloat4(c_parameters.resistanceArray.read(cell)) };
				const float vegetation = fmaxf(resistance.y, 0.f);
				const float abrasionScale{ c_parameters.deltaTime * windSpeed * (1.0f - vegetation) };

				const float vegetationFactor = (terrain.y > c_parameters.abrasionThreshold ? (0.4f) : (terrain.z > c_parameters.abrasionThreshold ? (0.5f) : 0.6f));
				const float waterProbability = (terrain.w > 1e-5f ? .01f : 1.f);
				const float depositionProbability = fminf(saltationMoistureFactor * fmaxf(resistance.x, (1.0f - vegetationFactor) + vegetation * vegetationFactor), waterProbability);

				const float new_slab = slab * (1.f - depositionProbability);
				const float waterFactor = (terrain.w > 1e-5f ? 0.f : 1.f);
				const float abrasion{ (terrain.y + terrain.z) < c_parameters.abrasionThreshold && new_slab > 0.f ? waterFactor * c_parameters.abrasionStrength * (1.0f - resistance.z) * abrasionScale * (1.f - depositionProbability) : 0.0f };

				const float soilAbrasion{ terrain.y < c_parameters.abrasionThreshold && new_slab > 0.f ? waterFactor * fminf(abrasionMoistureFactor * c_parameters.soilAbrasionStrength * abrasionScale * (1.f - depositionProbability), terrain.z) : 0.0f };

				terrain.y +=  abrasion + soilAbrasion;
				terrain.x -=  abrasion;
				terrain.z -= soilAbrasion;

				terrain.y += slab * depositionProbability;

				c_parameters.terrainArray.write(cell, toHalf4(terrain));
				c_parameters.slabBuffer[cellIndex] = __float2half(slab * (1.f - depositionProbability)); // write updated advectedSlabBuffer back to slabBuffer (ping-pong)
				t_advectedSlabBuffer[cellIndex] = __float2half(waterFactor * slab * (1.f - vegetation) * abrasionMoistureFactor); // Used in Reptation as slabBuffer
			}
		}
	}

	void continuousSaltation(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters)
	{
		setupContinuousSaltationKernel << <t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (t_launchParameters.tmpBuffer);

		if (t_launchParameters.saltationMode == SaltationMode::Backward) {
			if (t_launchParameters.useBilinear) {
			continuousBackwardSaltationKernel<true> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.tmpBuffer);
			}
			else {
				continuousBackwardSaltationKernel<false> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.tmpBuffer);
			}
		}
		else {
			if (t_launchParameters.useBilinear) {
			continuousSaltationKernel<true> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.tmpBuffer);
			}
			else {
				continuousSaltationKernel<false> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.tmpBuffer);
			}
		}

		finishContinuousSaltationKernel << <t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (t_launchParameters.tmpBuffer);
	}

}

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

	__global__ void setupContinuousSaltationKernel(Array2D<float4> t_terrainArray, const Array2D<float2> t_windArray, Array2D<float4> t_resistanceArray, Buffer<float> t_slabBuffer, Buffer<float> t_advectedSlabBuffer, Buffer<float> advectedAirMoistureBuffer)
	{
		const int2 index{ getGlobalIndex2D() };
		const int2 stride{ getGridStride2D() };

		int2 cell;

		for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
		{
			for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
			{
				float4 terrain{ t_terrainArray.read(cell) };

				const float2 windVelocity{ t_windArray.read(cell) };
				const float windSpeed{ length(windVelocity) };

				const float4 resistance{ t_resistanceArray.read(cell) };
				const float saltationScale{ (terrain.w > 1e-5f ? 0.f : 1.f) * (1.0f - resistance.x) * (1.0f - fmaxf(resistance.y, 0.f)) };

				// TODO: lower saltation when cell is wet
				//const float scale{ windSpeed * c_parameters.deltaTime };

				const float saltation{ fminf(c_parameters.saltationStrength * saltationScale, terrain.y) };

				terrain.y -= saltation;
				t_terrainArray.write(cell, terrain);

				const int cellIndex{ getCellIndex(cell) };
				const float slab{ saltation };

				t_slabBuffer[cellIndex] += slab;
				t_advectedSlabBuffer[cellIndex] = 0.0f;
				advectedAirMoistureBuffer[cellIndex] = 0.0f;
			}
		}
	}

	template <bool TUseBilinear>
	__global__ void continuousSaltationKernel(const Array2D<float2> t_windArray, Buffer<float> t_slabBuffer, Buffer<float> t_advectedSlabBuffer, Buffer<float> airMoistureBuffer, Buffer<float> advectedAirMoistureBuffer)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int cellIndex{ getCellIndex(cell) };
		const float slab{ t_slabBuffer[cellIndex] };
		const float moisture{ airMoistureBuffer[cellIndex] };

		const float2 windVelocity{ t_windArray.read(cell) };

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
							atomicAdd(t_advectedSlabBuffer + getCellIndex(getWrappedCell(int2{ x,y })), weight * slab);
							atomicAdd(advectedAirMoistureBuffer + getCellIndex(getWrappedCell(int2{ x,y })), weight * moisture);
						}
					}
				}
			}
			else {
				const int2 nextCell{ getNearestCell(nextPosition) };
				atomicAdd(t_advectedSlabBuffer + getCellIndex(getWrappedCell(nextCell)), slab);
				atomicAdd(advectedAirMoistureBuffer + getCellIndex(getWrappedCell(nextCell)), moisture);
			}
		}
	}

	template <bool TUseBilinear>
	__global__ void continuousBackwardSaltationKernel(const Array2D<float2> t_windArray, Buffer<float> t_slabBuffer, Buffer<float> t_advectedSlabBuffer, Buffer<float> airMoistureBuffer, Buffer<float> advectedAirMoistureBuffer)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int cellIndex{ getCellIndex(cell) };
		float slab{ 0.f };
		float moisture{ 0.f };

		const float2 windVelocity{ t_windArray.read(cell) };

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
						slab += t_slabBuffer[getCellIndex(getWrappedCell(int2{ x,y }))] * weight;
						moisture += airMoistureBuffer[getCellIndex(getWrappedCell(int2{ x,y }))] * weight;
					}
				}
			}
		}
		else {
			const int2 nextCell{ getNearestCell(nextPosition) };
			slab += t_slabBuffer[getCellIndex(getWrappedCell(nextCell))];
			moisture += airMoistureBuffer[getCellIndex(getWrappedCell(nextCell))];
		}


		t_advectedSlabBuffer[cellIndex] = slab;
		advectedAirMoistureBuffer[cellIndex] = moisture;
	}

	__global__ void finishContinuousSaltationKernel(Array2D<float4> t_terrainArray, const Array2D<float2> t_windArray, const Array2D<float4> t_resistanceArray, Buffer<float> t_slabBuffer, Buffer<float> t_advectedSlabBuffer, Buffer<float> airMoistureBuffer, Buffer<float> advectedAirMoistureBuffer)
	{
		const int2 index{ getGlobalIndex2D() };
		const int2 stride{ getGridStride2D() };

		int2 cell;

		for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
		{
			for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
			{
				const int cellIndex{ getCellIndex(cell) };

				float4 terrain{ t_terrainArray.read(cell) };
				const float slab{ t_advectedSlabBuffer[cellIndex] };
				const float moisture{ advectedAirMoistureBuffer[cellIndex] };

				const float windSpeed{ length(t_windArray.read(cell)) };

				const float4 resistance{ t_resistanceArray.read(cell) };
				const float vegetation = fmaxf(resistance.y, 0.f);
				const float abrasionScale{ c_parameters.deltaTime * windSpeed * (1.0f - vegetation) };
				// TODO: Depositionprob should be higher when cell is wet
				const float vegetationFactor = (terrain.y > c_parameters.abrasionThreshold ? 0.4f : (terrain.z > c_parameters.abrasionThreshold ? 0.5f : 0.6f));
				const float waterProbability = (terrain.w > 1e-5f ? 0.1f : 1.f);
				const float depositionProbability = fminf(fmaxf(resistance.x, (1.0f - vegetationFactor) + vegetation * vegetationFactor), waterProbability);

				const float new_slab = slab * (1.f - depositionProbability);
				const float waterFactor = (terrain.w > 1e-5f ? 0.f : 1.f);
				const float abrasion{ (terrain.y + terrain.z) < c_parameters.abrasionThreshold && new_slab > 0.f ? waterFactor * c_parameters.abrasionStrength * (1.0f - resistance.z) * abrasionScale * (1.f - depositionProbability) : 0.0f };
				// TODO: wet soil should be protected from abrasion
				const float soilAbrasion{ terrain.y < c_parameters.abrasionThreshold && new_slab > 0.f ? waterFactor * fminf(c_parameters.soilAbrasionStrength * abrasionScale * (1.f - depositionProbability), terrain.z) : 0.0f };

				const float temperature = 30.f - 0.01 * fmaxf(terrain.x + terrain.y + terrain.z + terrain.w, 0.f);
				const float airCapacity = 100000.f * (6.0328f * exp((17.1485f * temperature) / (234.69 + temperature))) / (461.52 * (temperature + 273.15));

				terrain.y +=  abrasion + soilAbrasion;
				terrain.x -=  abrasion;
				terrain.z -= soilAbrasion;
				//}
				terrain.y += slab * depositionProbability;
				const float deltaWater = fmaxf(moisture - airCapacity, 0.f);
				terrain.w += deltaWater;
				const float evaporation = 0.01f * fmaxf(airCapacity - (moisture - deltaWater), 0.f);
				terrain.w -= evaporation;
				t_terrainArray.write(cell, terrain);
				t_slabBuffer[cellIndex] = slab * (1.f - depositionProbability); // write updated advectedSlabBuffer back to slabBuffer (ping-pong)
				t_advectedSlabBuffer[cellIndex] = waterFactor * slab * (1.f - vegetation); // Used in Reptation as slabBuffer
				airMoistureBuffer[cellIndex] = moisture - deltaWater + evaporation;
			}
		}
	}

	void continuousSaltation(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters)
	{
		Buffer<float> advectedAirMoistureBuffer{ t_launchParameters.tmpBuffer + t_simulationParameters.cellCount };
		// TODO: implement Backward saltation (saltationMode)
		setupContinuousSaltationKernel << <t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.windArray, t_launchParameters.resistanceArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer, advectedAirMoistureBuffer);
		if (t_launchParameters.saltationMode == SaltationMode::Backward) {
			if (t_launchParameters.useBilinear) {
			continuousBackwardSaltationKernel<true> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.windArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer, t_launchParameters.airMoistureBuffer, advectedAirMoistureBuffer);
			}
			else {
				continuousBackwardSaltationKernel<false> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.windArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer, t_launchParameters.airMoistureBuffer, advectedAirMoistureBuffer);
			}
		}
		else {
			if (t_launchParameters.useBilinear) {
			continuousSaltationKernel<true> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.windArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer, t_launchParameters.airMoistureBuffer, advectedAirMoistureBuffer);
			}
			else {
				continuousSaltationKernel<false> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.windArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer, t_launchParameters.airMoistureBuffer, advectedAirMoistureBuffer);
			}
		}

		finishContinuousSaltationKernel << <t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.windArray, t_launchParameters.resistanceArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer, t_launchParameters.airMoistureBuffer, advectedAirMoistureBuffer);
	}

}

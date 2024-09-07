#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

	template<WindShadowMode Mode, bool TUseBilinear>
	__global__ void windShadowKernel()
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const float4 terrain{ c_parameters.terrainArray.read(cell) };
		float4 resistance{ c_parameters.resistanceArray.read(cell) };
		float2 windVelocity;
		float windSpeed;
		float2 windDirection;

		if constexpr (Mode == WindShadowMode::Linear)
		{
			windVelocity = c_parameters.windArray.read(cell);
			windSpeed = length(windVelocity);
			windDirection = windVelocity / (windSpeed + 1e-06f);
		}

		const float height{ terrain.x + terrain.y + terrain.z + terrain.w };
		float2 nextPosition{ make_float2(cell) + 0.5f };
		//nextPosition -= windDirection;
		float maxAngle{ 0.0f };

		for (float distance = c_parameters.gridScale; distance <= c_parameters.windShadowDistance; distance += c_parameters.gridScale)
		{
			if constexpr (Mode == WindShadowMode::Curved)
			{
				windVelocity = sampleLinearOrNearest<TUseBilinear>(c_parameters.windArray, nextPosition);
				windSpeed = length(windVelocity);
				windDirection = windVelocity / (windSpeed + 1e-06f);
			}

			nextPosition -= windDirection;

			const float4 nextTerrain{ sampleLinearOrNearest<TUseBilinear>(c_parameters.terrainArray, nextPosition) };
			const float nextHeight{ nextTerrain.x + nextTerrain.y + nextTerrain.z + nextTerrain.w };
			const float heightDifference{ nextHeight - height };
			const float angle{ heightDifference / distance };

			maxAngle = fmaxf(maxAngle, angle);

			if (maxAngle >= c_parameters.maxWindShadowAngle)
			{
				break;
			}
		}

		resistance.x = clamp((maxAngle - c_parameters.minWindShadowAngle) /
			(c_parameters.maxWindShadowAngle - c_parameters.minWindShadowAngle), 0.0f, 1.0f);

		c_parameters.resistanceArray.write(cell, resistance);
	}

	void windShadow(const LaunchParameters& t_launchParameters)
	{
		if (t_launchParameters.windShadowMode == WindShadowMode::Linear)
		{
			if (t_launchParameters.useBilinear)
				windShadowKernel<WindShadowMode::Linear, true> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > ();
			else
				windShadowKernel<WindShadowMode::Linear, false> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > ();
		}
		else
		{
			if (t_launchParameters.useBilinear)
				windShadowKernel<WindShadowMode::Curved, true> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > ();
			else
				windShadowKernel<WindShadowMode::Curved, false> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > ();
		}
	}

}

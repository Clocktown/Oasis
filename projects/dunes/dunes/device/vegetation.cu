#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes {

	__forceinline__ __device__ float getVegetationDensity(const Vegetation& veg, const float3& pos) {
		const float r2 = veg.radius * veg.radius;
		const float stem2 = veg.height.x * veg.height.x;
		const float root2 = veg.height.y * veg.height.y;
		const float3 covarStem{ 1.f / r2, 1.f / r2, 1.f / stem2 };
		const float3 covarRoot{ covarStem.x, covarStem.y, 1.f / root2 };
		const float3 covar = pos.z >= veg.pos.z ? covarStem : covarRoot;

		// Compute distance vector while considering toroidal boundaries
		float3 d = abs(pos - veg.pos);
		const float2 dims{ make_float2(c_parameters.gridSize) * c_parameters.gridScale };
		d.x = fminf(d.x, fabs(d.x - dims.x));
		d.y = fminf(d.y, fabs(d.y - dims.y));

		const float scale = 1.f; // peak vegetation density
		// gaussian distribution faded toward 0 at 2 * veg.radius
		return  fmaxf(scale * expf(-0.5f * dot(d, covar * d)) - 0.5 * (length(float2{pos.x - veg.pos.x, pos.y - veg.pos.y}) / veg.radius) * expf(-2.f), 0.f);

	}

	__global__ void rasterizeVegetation(Array2D<float2> t_terrainArray, Array2D<float4> t_resistanceArray /*, vegetationBuffer*/)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int2 vegCell{ 2048, 2048 };
		float2 terrain = t_terrainArray.read(vegCell);
		const float2 vegPos{ make_float2(vegCell) * c_parameters.gridScale };

		const Vegetation veg{
			0,
			float3{vegPos.x, vegPos.y, 0.f},
			float2{10.f, 10.f},
			float{512.f}
		};

		const Vegetation veg2{
			0,
			float3{vegPos.x * 0.5f, vegPos.y * 0.5f, 0.f},
			float2{20.f, 10.f},
			float{20.f}
		};


		const float2 position{ make_float2(cell) * c_parameters.gridScale };
		terrain = t_terrainArray.read(cell);
		const float3 pos{ position.x, position.y, terrain.x + terrain.y };

		float4 resistance = t_resistanceArray.read(cell);
		resistance.y = fminf(getVegetationDensity(veg, pos) + getVegetationDensity(veg2, pos), 1.f);
		t_resistanceArray.write(cell, resistance);
	}

	void vegetation(const LaunchParameters& t_launchParameters) {
		rasterizeVegetation << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.resistanceArray);
	}
}
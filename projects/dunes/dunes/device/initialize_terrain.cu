#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{
#define M_PI 3.1415926535897932384626433832795f

	// https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83
	__device__ float frand(float2 c) { return fract(sin(dot(c, float2{ 12.9898f, 78.233f })) * 43758.5453f); }

	__device__ float noise(float2 p, float freq)
	{
		float2 coords = p * freq;
		float2  ij = floorf(coords);
		float2  xy = coords - ij;

		//xy = 3.*xy*xy-2.*xy*xy*xy; // Alternative to cos
		xy = 0.5f * (1.f - cos(M_PI * xy));
		float a = frand((ij + float2{ 0.f, 0.f }));
		float b = frand((ij + float2{ 1.f, 0.f }));
		float c = frand((ij + float2{ 0.f, 1.f }));
		float d = frand((ij + float2{ 1.f, 1.f }));
		return bilerp(a, b, c, d, xy.x, xy.y);
	}

	__device__ float pNoise(float2 p, int res)
	{
		float persistance = .5f;
		float n = 0.f;
		float normK = 0.f;
		float f = 4.f;
		float amp = 1.f;
		for (int i = 0; i <= res; i++)
		{
			n += amp * noise(p, f);
			f *= 2.f;
			normK += amp;
			amp *= persistance;
		}
		float nf = n / normK;
		return nf * nf * nf * nf;
	}

	__device__ float seamless_pNoise(float2 off, float2 stretch, float2 uv, int res, float2 border)
	{
		float2 centered_uv_xy = 2.f * (uv - float2{ 0.5f, 0.5f });
		float2  limits = float2{ 1.f, 1.f } - 2.f * border;
		float2 distance = float2{ 0.f, 0.f };

		distance = max(abs(centered_uv_xy) - limits, float2{ 0.f, 0.f });

		centered_uv_xy = -1.f * sign(centered_uv_xy) * (limits + distance);

		distance /= 2.f * border;

		float2 xy_uv = 0.5f * centered_uv_xy + float2{ 0.5f, 0.5f };
		xy_uv = off + stretch * xy_uv;
		float2 base_uv = off + stretch * uv;

		float base_sample = pNoise(base_uv, res);

		if ((distance.x <= 0.f) && (distance.y <= 0.f))
		{
			return base_sample;
		}

		return bilerp(
			base_sample,
			pNoise(float2{ xy_uv.x, base_uv.y }, res),
			pNoise(float2{ base_uv.x, xy_uv.y }, res),
			pNoise(xy_uv, res),
			0.5f * smoothstep(0.f, 1.f, distance.x),
			0.5f * smoothstep(0.f, 1.f, distance.y)
		);
	}

	__global__ void initializeTerrainKernel(InitializationParameters t_initializationParameters)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const float2 uv = (make_float2(cell) + 0.5f) / make_float2(c_parameters.gridSize);

		const float4 curr_terrain = half4toFloat4(c_parameters.terrainArray.read(cell));
		const float4 curr_resistance = half4toFloat4(c_parameters.resistanceArray.read(cell));
		const float curr_moisture = __half2float(c_parameters.moistureArray.read(cell));

		const int indices[7]{
			(int)NoiseGenerationTarget::Bedrock,
			(int)NoiseGenerationTarget::Sand,
			(int)NoiseGenerationTarget::Vegetation,
			(int)NoiseGenerationTarget::AbrasionResistance,
			(int)NoiseGenerationTarget::Soil,
			(int)NoiseGenerationTarget::Water,
			(int)NoiseGenerationTarget::Moisture
		};
		float values[7] = { curr_terrain.x, curr_terrain.y, curr_resistance.y, curr_resistance.z, curr_terrain.z, curr_terrain.w, curr_moisture };

		for (int i = 0; i < 7; ++i) {
			const auto& params = t_initializationParameters.noiseGenerationParameters[indices[i]];
			values[i] = params.enabled ?
				(params.bias +
					(params.uniform_random ? params.scale * frand(params.offset + params.stretch * uv) :
						(params.flat ?
							0.0f :
							params.scale * seamless_pNoise(params.offset, params.stretch, uv, params.iters, params.border)))) :
				values[i];
		}

		values[2] = clamp(values[2], 0.f, 1.f);
		values[3] = clamp(values[3], 0.f, 1.f);
		values[1] = fmaxf(values[1], 0.f);
		values[4] = fmaxf(values[4], 0.f);
		values[5] = fmaxf(values[5] - (values[0] + values[1] + values[4]), 0.f);
		values[6] = fmaxf(values[6], 0.f);

		// Regular initialization
		const float4 terrain{ values[0], values[1], values[4], values[5]};
		c_parameters.terrainArray.write(cell, toHalf4(terrain));

		const float4 resistance{ 0.0f, values[2], values[3], 0.0f};
		c_parameters.resistanceArray.write(cell, toHalf4(resistance));

		const int idx = getCellIndex(cell);
		c_parameters.slabBuffer[idx] = CUDART_ZERO_FP16;
		c_parameters.moistureArray.write(cell, __float2half(values[6]));
        c_parameters.fluxArray.write(cell,
                                        half4 {half2 {CUDART_ZERO_FP16, CUDART_ZERO_FP16},
                                            half2 {CUDART_ZERO_FP16, CUDART_ZERO_FP16}});
        c_parameters.sedimentArray.write(cell, CUDART_ZERO_FP16);
	}

	__global__ void addSandForCoverageKernel(float amount)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		float4 curr_terrain = half4toFloat4(c_parameters.terrainArray.read(cell));

		curr_terrain.y += frand(make_float2(cell)) * 2.f * amount;
		curr_terrain.y = fmaxf(curr_terrain.y, 0.f);

		c_parameters.terrainArray.write(cell, toHalf4(curr_terrain));
	}

	__global__ void addSandCircleForCoverageKernel(int2 pos, int radius, float amount)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		float4 curr_terrain = half4toFloat4(c_parameters.terrainArray.read(cell));

		const float2 cellf{ make_float2(cell) };
		const float2 posf{ make_float2(pos) };
		float2 distance{ cellf.x - posf.x, cellf.y - posf.y };
		distance.x = fminf(fabsf(distance.x), fminf(fabsf(distance.x + c_parameters.gridSize.x), fabsf(distance.x - c_parameters.gridSize.x)));
		distance.y = fminf(fabsf(distance.y), fminf(fabsf(distance.y + c_parameters.gridSize.y), fabsf(distance.y - c_parameters.gridSize.y)));

		if (length(distance) <= radius) {
			curr_terrain.y += frand(make_float2(cell)) * 2.f * amount;;
		}


		c_parameters.terrainArray.write(cell, toHalf4(curr_terrain));
	}

	void initializeTerrain(const LaunchParameters& t_launchParameters, const InitializationParameters& t_initializationParameters)
	{
		initializeTerrainKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_initializationParameters);
	}

	void addSandForCoverage(const LaunchParameters& t_launchParameters, int2 res, bool uniform, int radius, float amount) {
		if (uniform) {
			addSandForCoverageKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (amount);
		}
		else {
			addSandCircleForCoverageKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (int2{ rand() % res.x, rand() % res.y }, radius, amount);
		}
	}

}

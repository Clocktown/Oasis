#pragma once
#include <cuda_runtime.h>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{
namespace random
{

CU_INLINE CU_HOST_DEVICE unsigned int pcg(unsigned int& seed) noexcept
{
	unsigned int value{ seed * 747796405u + 2891336453u };
	value = ((value >> ((value >> 28u) + 4u)) ^ value) * 277803737u;
	value = (value >> 22u) ^ value;

	seed = value;

	return value;
}

CU_INLINE CU_HOST_DEVICE uint2 pcg(uint2& seed) noexcept
{
	uint2 value{ seed * 1664525u + 1013904223u };
	value.x += value.y * 1664525u;
	value.y += value.x * 1664525u;

	value = value ^ (value >> 16u);

	value.x += value.y * 1664525u;
	value.y += value.x * 1664525u;

	value = value ^ (value >> 16u);

	seed = value;

	return value;
}

CU_INLINE CU_HOST_DEVICE uint3 pcg(uint3& seed) noexcept
{
	uint3 value{ seed * 1664525u + 1013904223u };

	value.x += value.y * value.z;
	value.y += value.z * value.x;
	value.z += value.x * value.y;

	value = value ^ (value >> 16u);

	value.x += value.y * value.z;
	value.y += value.z * value.x;
	value.z += value.x * value.y;

	seed = value;

	return value;
}

CU_INLINE CU_HOST_DEVICE uint4 pcg(uint4& seed) noexcept
{
	uint4 value{ seed * 1664525u + 1013904223u };

	value.x += value.y * value.w;
	value.y += value.z * value.x;
	value.z += value.x * value.y;
	value.w += value.y * value.z;

	value = value ^ (value >> 16u);

	value.x += value.y * value.w;
	value.y += value.z * value.x;
	value.z += value.x * value.y;
	value.w += value.y * value.z;

	seed = value;

	return value;
}

}
}

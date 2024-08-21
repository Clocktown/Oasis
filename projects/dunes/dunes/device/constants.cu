#include <dunes/core/simulation_parameters.hpp>
#include <cuda_runtime.h>

#define SQRT2 1.414213562f
#define RSQRT2 0.707106782f

namespace dunes
{

__constant__ SimulationParameters c_parameters{};

__constant__ int2 c_offsets[8]{ int2{ 1, 0 }, int2{ 1, 1 }, int2{ 0, 1 }, int2{ -1, 1 },
                                int2{ -1, 0 }, int2{ -1, -1 }, int2{ 0, -1 }, int2{ 1, -1 } };

__constant__ float c_distances[8]{ 1.0f, SQRT2, 1.0f, SQRT2, 1.0f, SQRT2, 1.0f, SQRT2 };
__constant__ float c_rDistances[8]{ 1.0f, RSQRT2, 1.0f, RSQRT2, 1.0f, RSQRT2, 1.0f, RSQRT2 };
__constant__ VegetationType c_vegTypes[1]{ 
    {
        20.f,
        0.1f,
        60.f,
        {2.f, 1.f},
        0.1f,
        1.f,
        0.1f,
        1.f,
        {0.2f, 0.8f},
        1.f,
        0.2f
    } 
};
}

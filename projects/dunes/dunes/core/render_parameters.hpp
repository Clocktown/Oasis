#pragma once

#include <glm/glm.hpp>
#include <array>
#include <vector>

namespace dunes
{

struct RenderParameters
{
	glm::vec4 sandColor{ 0.9f, 0.8f, 0.6f, 1.0f };
	glm::vec4 bedrockColor{ 0.5f, 0.5f, 0.5f, 1.0f };
	glm::vec4 windShadowColor{ 1.0f, 0.25f, 0.25f, 1.0f };
	glm::vec4 vegetationColor{ 0.25f, 0.75f, 0.25f, 1.0f };
	glm::vec4 soilColor{ 0.55f, 0.35f, 0.2f, 1.0f };
	glm::vec4 waterColor{ 0.1f, 0.4f, 1.0f, 1.0f };
	glm::vec4 humusColor{ 0.35f, 0.3f, 0.15f, 1.f };
};

}

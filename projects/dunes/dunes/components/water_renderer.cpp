#include "water_renderer.hpp"
#include <dunes/core/water.hpp>
#include <sthe/core/custom_material.hpp>
#include <memory>
#include <vector>

namespace dunes
{
using namespace sthe;
// Constructors
WaterRenderer::WaterRenderer(const std::shared_ptr<Water>& t_water, const std::shared_ptr<CustomMaterial>& t_material, const std::shared_ptr<CustomMaterial>& t_rimMaterial) :
	m_water{ t_water },
	m_material{ t_material },
	m_rimMaterial{ t_rimMaterial }
{

}

// Setters
void WaterRenderer::setWater(const std::shared_ptr<Water>& t_water)
{
	m_water = t_water;
}

void WaterRenderer::setMaterial(const std::shared_ptr<CustomMaterial>& t_material)
{
	m_material = t_material;
}

void WaterRenderer::setRimMaterial(const std::shared_ptr<CustomMaterial>& t_material)
{
	m_rimMaterial = t_material;
}

// Getters
const std::shared_ptr<Water>& WaterRenderer::getWater() const
{
	return m_water;
}

const std::shared_ptr<CustomMaterial>& WaterRenderer::getMaterial() const
{
	return m_material;
}

const std::shared_ptr<CustomMaterial>& WaterRenderer::getRimMaterial() const
{
	return m_rimMaterial;
}

bool WaterRenderer::hasWater() const
{
	return m_water != nullptr;
}

bool WaterRenderer::hasMaterial() const
{
	return m_material != nullptr;
}

bool WaterRenderer::hasRimMaterial() const
{
	return m_rimMaterial != nullptr;
}

}

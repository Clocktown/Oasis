#include "terrain_renderer.hpp"
#include <sthe/core/terrain.hpp>
#include <sthe/core/custom_material.hpp>
#include <memory>
#include <vector>

namespace sthe
{

// Constructors
TerrainRenderer::TerrainRenderer(const std::shared_ptr<Terrain>& t_terrain, const std::shared_ptr<CustomMaterial>& t_material, const std::shared_ptr<CustomMaterial>& t_rimMaterial) :
	m_terrain{ t_terrain },
	m_material{ t_material },
	m_rimMaterial{ t_rimMaterial }
{

}

// Setters
void TerrainRenderer::setTerrain(const std::shared_ptr<Terrain>& t_terrain)
{
	m_terrain = t_terrain;
}

void TerrainRenderer::setMaterial(const std::shared_ptr<CustomMaterial>& t_material)
{
	m_material = t_material;
}

void TerrainRenderer::setRimMaterial(const std::shared_ptr<CustomMaterial>& t_material)
{
	m_rimMaterial = t_material;
}

// Getters
const std::shared_ptr<Terrain>& TerrainRenderer::getTerrain() const
{
	return m_terrain;
}

const std::shared_ptr<CustomMaterial>& TerrainRenderer::getMaterial() const
{
	return m_material;
}

const std::shared_ptr<CustomMaterial>& TerrainRenderer::getRimMaterial() const
{
	return m_rimMaterial;
}

bool TerrainRenderer::hasTerrain() const
{
	return m_terrain != nullptr;
}

bool TerrainRenderer::hasMaterial() const
{
	return m_material != nullptr;
}

bool TerrainRenderer::hasRimMaterial() const
{
	return m_rimMaterial != nullptr;
}

}

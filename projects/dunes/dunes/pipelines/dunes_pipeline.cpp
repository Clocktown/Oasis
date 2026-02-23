#include <dunes/core/water.hpp>
#include <dunes/core/ui.hpp>
#include "dunes_pipeline.hpp"
#include <dunes/components/water_renderer.hpp>
#include <sthe/config/debug.hpp>
#include <sthe/config/binding.hpp>
#include <dunes/util/io.hpp>
#include <sthe/util/io.hpp>
#include <sthe/core/application.hpp>
#include <sthe/core/scene.hpp>
#include <sthe/core/environment.hpp>
#include <sthe/core/mesh.hpp>
#include <sthe/core/sub_mesh.hpp>
#include <sthe/core/material.hpp>
#include <sthe/core/terrain.hpp>
#include <sthe/core/terrain_layer.hpp>
#include <sthe/components/transform.hpp>
#include <sthe/components/light.hpp>
#include <sthe/components/mesh_renderer.hpp>
#include <sthe/components/terrain_renderer.hpp>
#include <sthe/core/application.hpp>
#include <sthe/gl/program.hpp>
#include <sthe/gl/buffer.hpp>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <entt/entt.hpp>
#include <memory>
#include <array>
#include <vector>

namespace dunes
{
using namespace sthe;
// Constructors
DunesPipeline::DunesPipeline() :
	m_meshProgram{ std::make_shared<gl::Program>() },
	m_terrainProgram{ std::make_shared<gl::Program>() },
	m_pipelineBuffer{ std::make_shared<gl::Buffer>(static_cast<int>(sizeof(uniform::DunesPipeline)), 1) },
	m_terrainFrameBuffer{std::make_shared<gl::Framebuffer>(1, 1) },
	m_waterFrameBuffer{std::make_shared<gl::Framebuffer>(1, 1) },
	m_terrainDiffuseMap{ std::make_shared<sthe::gl::Texture2D>() },
	m_terrainPositionMap{ std::make_shared<sthe::gl::Texture2D>() },
	m_depthMap{ std::make_shared<sthe::gl::Texture2D>() },
	m_waterPositionOffsetMap{ std::make_shared<sthe::gl::Texture2D>() },
	m_waterDiffuseMap{ std::make_shared<sthe::gl::Texture2D>() },
	m_screenMaterial{ std::make_shared<sthe::CustomMaterial>() },
	m_screenProgram{ std::make_shared<sthe::gl::Program>() },
	m_zPassProgram{ std::make_shared<sthe::gl::Program>() },
	m_vertexArray{ std::make_shared<sthe::gl::VertexArray>() }
{
	setupFramebuffers(1, 1);

	m_screenProgram->attachShader(sthe::gl::Shader{ GL_VERTEX_SHADER, dunes::getShaderPath() + "screen/sfq.vert" });
	m_screenProgram->attachShader(sthe::gl::Shader{ GL_FRAGMENT_SHADER, dunes::getShaderPath() + "screen/sfq.frag" });
	m_screenProgram->link();

	m_zPassProgram->attachShader(sthe::gl::Shader{ GL_VERTEX_SHADER, dunes::getShaderPath() + "zpass/zpass.vert" });
	m_zPassProgram->attachShader(sthe::gl::Shader{ GL_FRAGMENT_SHADER, dunes::getShaderPath() + "zpass/zpass.frag" });
	m_zPassProgram->link();

	m_screenMaterial->setProgram(m_screenProgram);
	m_screenMaterial->setTexture(0, m_terrainDiffuseMap);
	m_screenMaterial->setTexture(1, m_terrainPositionMap);
	m_screenMaterial->setTexture(4, m_waterDiffuseMap);
	m_screenMaterial->setTexture(3, m_waterPositionOffsetMap);

	m_meshProgram->attachShader(gl::Shader{ GL_VERTEX_SHADER, sthe::getShaderPath() + "mesh/phong.vert" });
    m_meshProgram->attachShader(gl::Shader{ GL_FRAGMENT_SHADER, sthe::getShaderPath() + "mesh/phong.frag" });
    m_meshProgram->link();

	m_terrainProgram->setPatchVertexCount(4);
	m_terrainProgram->attachShader(gl::Shader{ GL_VERTEX_SHADER, sthe::getShaderPath() + "terrain/phong.vert" });
	m_terrainProgram->attachShader(gl::Shader{ GL_TESS_CONTROL_SHADER, sthe::getShaderPath() + "terrain/phong.tesc" });
	m_terrainProgram->attachShader(gl::Shader{ GL_TESS_EVALUATION_SHADER, sthe::getShaderPath() + "terrain/phong.tese" });
	m_terrainProgram->attachShader(gl::Shader{ GL_FRAGMENT_SHADER, sthe::getShaderPath() + "terrain/phong.frag" });
	m_terrainProgram->link();
}

// Functionality

void DunesPipeline::setupFramebuffers(int width, int height) {
	m_terrainFrameBuffer->resize(width, height);
	m_waterFrameBuffer->resize(width, height);
	m_terrainDiffuseMap->reinitialize(width, height, GL_RGBA8, false);
	m_terrainPositionMap->reinitialize(width, height, GL_RGBA32F, false);
	m_depthMap->reinitialize(width, height, GL_DEPTH_COMPONENT24, false);
	m_waterDiffuseMap->reinitialize(width, height, GL_RGBA32F, false);
	m_waterPositionOffsetMap->reinitialize(width, height, GL_RGBA32F, false);
	m_terrainFrameBuffer->attachTexture(GL_COLOR_ATTACHMENT0, *m_terrainDiffuseMap);
	m_terrainFrameBuffer->attachTexture(GL_COLOR_ATTACHMENT1, *m_terrainPositionMap);
	m_terrainFrameBuffer->attachTexture(GL_DEPTH_ATTACHMENT, *m_depthMap);
	m_waterFrameBuffer->attachTexture(GL_COLOR_ATTACHMENT0, *m_waterDiffuseMap);
	m_waterFrameBuffer->attachTexture(GL_COLOR_ATTACHMENT1, *m_waterPositionOffsetMap);
	m_waterFrameBuffer->attachTexture(GL_DEPTH_ATTACHMENT, *m_depthMap);
}

void DunesPipeline::use()
{
	GL_CHECK_ERROR(glEnable(GL_DEPTH_TEST));
	GL_CHECK_ERROR(glEnable(GL_SCISSOR_TEST));
	GL_CHECK_ERROR(glEnable(GL_CULL_FACE));
}

void DunesPipeline::disuse()
{
	GL_CHECK_ERROR(glDisable(GL_DEPTH_TEST));
	GL_CHECK_ERROR(glDisable(GL_SCISSOR_TEST));
	GL_CHECK_ERROR(glDisable(GL_CULL_FACE));
}

void DunesPipeline::render(const Scene& t_scene, const Camera& t_camera)
{
	use();
	setup(t_scene, t_camera);

	m_terrainFrameBuffer->bind();


	GL_CHECK_ERROR(glScissor(0, 0, m_data.resolution.x, m_data.resolution.y));
	GL_CHECK_ERROR(glViewport(0, 0, m_data.resolution.x, m_data.resolution.y));

	m_terrainFrameBuffer->enableDrawBuffer(GL_COLOR_ATTACHMENT0);
	const glm::vec4& clearColor{ t_camera.getClearColor() };
	GL_CHECK_ERROR(glClearColor(clearColor.r, clearColor.g, clearColor.b, 0.f));
	GL_CHECK_ERROR(glClear(t_camera.getClearMask()));
	m_terrainFrameBuffer->enableDrawBuffer(GL_COLOR_ATTACHMENT1);
	glClearColor(0.f, 0.f, 0.f, 0.f);
	glClear(GL_COLOR_BUFFER_BIT);

	m_terrainFrameBuffer->enableDrawBuffers({ GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 });
	glDisable(GL_CULL_FACE);
	meshRendererPass(t_scene);
	glEnable(GL_CULL_FACE);
	terrainRendererPass(t_scene);
	//m_terrainFrameBuffer->disableDrawBuffers();
	//meshRendererZPass(t_scene);
	//m_terrainFrameBuffer->enableDrawBuffers({ GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 });
	//glDepthFunc(GL_LEQUAL);

	//glDepthFunc(GL_LESS);
	m_terrainFrameBuffer->disableDrawBuffers();

	m_waterFrameBuffer->bind();
	m_waterFrameBuffer->enableDrawBuffers({ GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 });
	glClear(GL_COLOR_BUFFER_BIT);

	waterRendererPass(t_scene);

	gl::DefaultFramebuffer::bind();
	GL_CHECK_ERROR(glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a));
	GL_CHECK_ERROR(glClear(t_camera.getClearMask()));
	glDisable(GL_DEPTH_TEST);
	m_screenMaterial->bind();
	m_screenProgram->use();

	m_vertexArray->bind();

	GL_CHECK_ERROR(glDrawArrays(GL_TRIANGLES, 0, 3));
	disuse();
}

void DunesPipeline::setup(const Scene& t_scene, const Camera& t_camera)
{
	const Application& application{ getApplication() };
	const Environment& environment{ t_scene.getEnvironment() };

	if (t_camera.getResolution().x != m_terrainFrameBuffer->getWidth() || t_camera.getResolution().y != m_terrainFrameBuffer->getHeight()) {
		setupFramebuffers(t_camera.getResolution().x, t_camera.getResolution().y);
	}

	m_data.time = application.getTime();
	m_data.deltaTime = application.getDeltaTime();
	m_data.resolution = t_camera.getResolution();
	m_data.projectionMatrix = t_camera.getProjectionMatrix();
	m_data.viewMatrix = t_camera.getViewMatrix();
	m_data.inverseViewMatrix = t_camera.getInverseViewMatrix();
	m_data.viewProjectionMatrix = m_data.projectionMatrix * m_data.viewMatrix;
	m_data.environment.ambientColor = environment.getAmbientColor();
	m_data.environment.ambientIntensity = environment.getAmbientIntensity();
	m_data.environment.fogColor = environment.getFogColor();
	m_data.environment.fogDensity = environment.getFogDensity();
	m_data.environment.fogMode = static_cast<unsigned int>(environment.getFogMode());
	m_data.environment.fogStart = environment.getFogStart();
	m_data.environment.fogEnd = environment.getFogEnd();
	m_data.environment.lightCount = 0;

	const auto lights{ t_scene.getComponents<Transform, Light>() };

	for (const entt::entity entity : lights)
	{
		const auto [transform, light] { lights.get<const Transform, const Light>(entity) };

		sthe::uniform::Light& lightData{ m_data.environment.lights[m_data.environment.lightCount++] };
		lightData.position = transform.getPosition();
		lightData.type = static_cast<unsigned int>(light.getType());
		lightData.color = light.getColor();
		lightData.intensity = light.getIntensity();
		lightData.attenuation = light.getAttenuation();
		lightData.range = light.getRange();
		lightData.direction = transform.getForward();
		lightData.spotOuterCutOff = glm::cos(glm::radians(light.getSpotAngle()));
		lightData.spotInnerCutOff = glm::cos(glm::radians((1.0f - light.getSpotBlend()) * light.getSpotAngle()));

		if (m_data.environment.lightCount >= m_data.environment.lights.size())
		{
			break;
		}
	}

	m_pipelineBuffer->bind(GL_UNIFORM_BUFFER, STHE_UNIFORM_BUFFER_PIPELINE);
	m_pipelineBuffer->upload(reinterpret_cast<char*>(&m_data), static_cast<int>(offsetof(uniform::DunesPipeline, environment.lights)) +
		                     m_data.environment.lightCount * static_cast<int>(sizeof(sthe::uniform::Light)));
}

void DunesPipeline::meshRendererPass(const Scene& t_scene)
{
	const dunes::UI& ui = t_scene.getComponents<dunes::UI>().get<const dunes::UI>(t_scene.getComponents<dunes::UI>().front());
	if (!ui.render_vegetation) {
		return;
	}
	const Material* activeMaterial{ nullptr };
	const gl::Program* activeProgram{ nullptr };
	
	const auto meshRenderers{ t_scene.getComponents<Transform, MeshRenderer>() };
	
	for (const entt::entity entity : meshRenderers)
	{
		const MeshRenderer& meshRenderer{ meshRenderers.get<const MeshRenderer>(entity) };

		if (meshRenderer.hasMesh() && meshRenderer.getMaterialCount() > 0)
		{
			const Transform& transform{ meshRenderers.get<const Transform>(entity) };

			m_data.modelMatrix = transform.getModelMatrix();
			m_data.inverseModelMatrix = transform.getInverseModelMatrix();
			m_data.modelViewMatrix = m_data.viewMatrix * m_data.modelMatrix;
			m_data.inverseModelViewMatrix = m_data.inverseModelMatrix * m_data.inverseViewMatrix;
			m_data.userID = meshRenderer.getUserID();
			m_pipelineBuffer->upload(reinterpret_cast<char*>(&m_data.modelMatrix), static_cast<int>(offsetof(uniform::DunesPipeline, modelMatrix)), 4 * sizeof(glm::mat4) + sizeof(glm::ivec4));

			const Mesh& mesh{ *meshRenderer.getMesh() };
			mesh.bind();

			const std::vector<std::shared_ptr<Material>>& materials{ meshRenderer.getMaterials() };

			for (int i{ 0 }, j{ 0 }; i < mesh.getSubMeshCount(); ++i, j = std::min(j + 1, meshRenderer.getMaterialCount() - 1))
			{
				if (activeMaterial != materials[j].get())
				{
					activeMaterial = materials[j].get();
					activeMaterial->bind();

					const gl::Program* const program{ activeMaterial->hasProgram() ? activeMaterial->getProgram().get() :
																					 m_meshProgram.get() };

					if (activeProgram != program)
					{
						activeProgram = program;
						activeProgram->use();
					}
					
					sthe::uniform::Material& materialData{ m_data.material };
					materialData.diffuseColor = activeMaterial->getDiffuseColor();
					materialData.opacity = activeMaterial->getOpacity();
					materialData.specularColor = activeMaterial->getSpecularColor();
					materialData.specularIntensity = activeMaterial->getSpecularIntensity();
					materialData.shininess = activeMaterial->getShininess();
					materialData.hasDiffuseMap = activeMaterial->hasDiffuseMap();
					materialData.hasNormalMap = activeMaterial->hasNormalMap();
					materialData.custom = activeMaterial->getCustom();

					m_pipelineBuffer->upload(reinterpret_cast<char*>(&materialData), static_cast<int>(offsetof(uniform::DunesPipeline, material)), sizeof(sthe::uniform::Material));
				}

				const SubMesh& subMesh{ mesh.getSubMesh(i) };
				const long long int offset{ subMesh.getFirstIndex() * static_cast<long long int>(sizeof(int)) };

				GL_CHECK_ERROR(glDrawElementsInstancedBaseInstance(subMesh.getDrawMode(), subMesh.getIndexCount(), GL_UNSIGNED_INT, reinterpret_cast<void*>(offset),
							   meshRenderer.getInstanceCount(), static_cast<GLuint>(meshRenderer.getBaseInstance())));
			}
		}
	}
}

void DunesPipeline::meshRendererZPass(const Scene& t_scene)
{
	const gl::Program* activeProgram{ m_zPassProgram.get() };
	activeProgram->use();
	
	const auto meshRenderers{ t_scene.getComponents<Transform, MeshRenderer>() };
	
	for (const entt::entity entity : meshRenderers)
	{
		const MeshRenderer& meshRenderer{ meshRenderers.get<const MeshRenderer>(entity) };

		if (meshRenderer.hasMesh() && meshRenderer.getMaterialCount() > 0)
		{
			const Transform& transform{ meshRenderers.get<const Transform>(entity) };

			m_data.modelMatrix = transform.getModelMatrix();
			m_data.inverseModelMatrix = transform.getInverseModelMatrix();
			m_data.modelViewMatrix = m_data.viewMatrix * m_data.modelMatrix;
			m_data.inverseModelViewMatrix = m_data.inverseModelMatrix * m_data.inverseViewMatrix;
			m_data.userID = meshRenderer.getUserID();
			m_pipelineBuffer->upload(reinterpret_cast<char*>(&m_data.modelMatrix), static_cast<int>(offsetof(uniform::DunesPipeline, modelMatrix)), 4 * sizeof(glm::mat4) + sizeof(glm::ivec4));

			const Mesh& mesh{ *meshRenderer.getMesh() };
			mesh.bind();

			const std::vector<std::shared_ptr<Material>>& materials{ meshRenderer.getMaterials() };

			for (int i{ 0 }, j{ 0 }; i < mesh.getSubMeshCount(); ++i, j = std::min(j + 1, meshRenderer.getMaterialCount() - 1))
			{
				const SubMesh& subMesh{ mesh.getSubMesh(i) };
				const long long int offset{ subMesh.getFirstIndex() * static_cast<long long int>(sizeof(int)) };

				GL_CHECK_ERROR(glDrawElementsInstancedBaseInstance(subMesh.getDrawMode(), subMesh.getIndexCount(), GL_UNSIGNED_INT, reinterpret_cast<void*>(offset),
							   meshRenderer.getInstanceCount(), static_cast<GLuint>(meshRenderer.getBaseInstance())));
			}
		}
	}
}


void DunesPipeline::terrainRendererPass(const Scene& t_scene)
{
	const CustomMaterial* activeMaterial{ nullptr };
	const gl::Program* activeProgram{ nullptr };

	const auto terrainRenderers{ t_scene.getComponents<Transform, TerrainRenderer>() };

	for (const entt::entity entity : terrainRenderers)
	{
		const TerrainRenderer& terrainRenderer{ terrainRenderers.get<const TerrainRenderer>(entity) };

		if (terrainRenderer.hasTerrain() && terrainRenderer.hasMaterial())
		{
			const Transform& transform{ terrainRenderers.get<const Transform>(entity) };

			m_data.modelMatrix = transform.getModelMatrix();
			m_data.inverseModelMatrix = transform.getInverseModelMatrix();
			m_data.modelViewMatrix = m_data.viewMatrix * m_data.modelMatrix;
			m_data.inverseModelViewMatrix = m_data.inverseModelMatrix * m_data.inverseViewMatrix;
			m_pipelineBuffer->upload(reinterpret_cast<char*>(&m_data.modelMatrix), static_cast<int>(offsetof(uniform::DunesPipeline, modelMatrix)), 4 * sizeof(glm::mat4));

			const CustomMaterial* const material{ terrainRenderer.getMaterial().get() };

			if (activeMaterial != material)
			{
				activeMaterial = material;
				activeMaterial->bind();
			}

			const gl::Program* const program{ activeMaterial->hasProgram() ? activeMaterial->getProgram().get() :
																		     m_terrainProgram.get() };

			if (activeProgram != program)
			{
				activeProgram = program;
				activeProgram->use();
			}

			const Terrain& terrain{ *terrainRenderer.getTerrain() };
			terrain.bind();

			sthe::uniform::Terrain& terrainData{ m_data.terrain };
			terrainData.gridSize = terrain.getGridSize();
			terrainData.gridScale = terrain.getGridScale();
			terrainData.heightScale = terrain.getHeightScale();
			terrainData.tesselationLevel = terrain.getTesselationLevel();
			terrainData.hasHeightMap = terrain.hasHeightMap();
			terrainData.hasAlphaMap = terrain.hasAlphaMap();
			terrainData.layerCount = 0;

			for (const auto& terrainLayer : terrain.getLayers())
			{
				sthe::uniform::TerrainLayer& terrainLayerData{ terrainData.layers[terrainData.layerCount++] };
				terrainLayerData.diffuseColor = terrainLayer->getDiffuseColor();
				terrainLayerData.specularIntensity = terrainLayer->getSpecularIntensity();
				terrainLayerData.specularColor = terrainLayer->getSpecularColor();
				terrainLayerData.shininess = terrainLayer->getShininess();
				terrainLayerData.hasDiffuseMap = terrainLayer->hasDiffuseMap();
			}

			m_pipelineBuffer->upload(reinterpret_cast<char*>(&terrainData), static_cast<int>(offsetof(uniform::DunesPipeline, terrain)), sizeof(sthe::uniform::Terrain));
			
			const glm::ivec2 subDivision = terrainData.gridSize / terrainData.tesselationLevel;

			GL_CHECK_ERROR(glDrawArraysInstanced(GL_PATCHES, 0, 4, subDivision.x * subDivision.y));

			const CustomMaterial* const rimMaterial{ terrainRenderer.getRimMaterial().get() };

			if (activeMaterial != rimMaterial)
			{
				activeMaterial = rimMaterial;
				activeMaterial->bind();
			}

			if (!activeMaterial->hasProgram()) {
				continue;
			}

			const gl::Program* const rimProgram{ activeMaterial->getProgram().get() };

			if (activeProgram != rimProgram)
			{
				activeProgram = rimProgram;
				activeProgram->use();
			}

			GL_CHECK_ERROR(glDrawArraysInstanced(GL_PATCHES, 0, 4, 2 * subDivision.x + 2 * subDivision.y));
		}
	}
}

void DunesPipeline::waterRendererPass(const Scene& t_scene)
{
	const CustomMaterial* activeMaterial{ nullptr };
	const gl::Program* activeProgram{ nullptr };

	const auto waterRenderers{ t_scene.getComponents<Transform, WaterRenderer>() };

	for (const entt::entity entity : waterRenderers)
	{
		const WaterRenderer& waterRenderer{ waterRenderers.get<const WaterRenderer>(entity) };

		if (waterRenderer.hasWater() && waterRenderer.hasMaterial())
		{
			const CustomMaterial* const material{ waterRenderer.getMaterial().get() };

			if (activeMaterial != material)
			{
				activeMaterial = material;
				activeMaterial->bind();
			}

			if (!activeMaterial->hasProgram()) {
				continue;
			}

			const Transform& transform{ waterRenderers.get<const Transform>(entity) };

			m_data.modelMatrix = transform.getModelMatrix();
			m_data.inverseModelMatrix = transform.getInverseModelMatrix();
			m_data.modelViewMatrix = m_data.viewMatrix * m_data.modelMatrix;
			m_data.inverseModelViewMatrix = m_data.inverseModelMatrix * m_data.inverseViewMatrix;
			m_pipelineBuffer->upload(reinterpret_cast<char*>(&m_data.modelMatrix), static_cast<int>(offsetof(uniform::DunesPipeline, modelMatrix)), 4 * sizeof(glm::mat4));

			const gl::Program* const program{ activeMaterial->getProgram().get() };

			if (activeProgram != program)
			{
				activeProgram = program;
				activeProgram->use();
			}

			const Water& water{ *waterRenderer.getWater() };
			water.bind();

			dunes::uniform::Water& waterData{ m_data.water };
			waterData.gridSize = water.getGridSize();
			waterData.gridScale = water.getGridScale();
			waterData.heightScale = water.getHeightScale();
			waterData.tesselationLevel = water.getTesselationLevel();
			waterData.hasHeightMap = water.hasHeightMap();

			m_pipelineBuffer->upload(reinterpret_cast<char*>(&waterData), static_cast<int>(offsetof(uniform::DunesPipeline, water)), sizeof(dunes::uniform::Water));
			
			const glm::ivec2 subDivision = waterData.gridSize / waterData.tesselationLevel;

			GL_CHECK_ERROR(glDrawArraysInstanced(GL_PATCHES, 0, 4, subDivision.x * subDivision.y));

			const CustomMaterial* const rimMaterial{ waterRenderer.getRimMaterial().get() };

			if (activeMaterial != rimMaterial)
			{
				activeMaterial = rimMaterial;
				activeMaterial->bind();
			}

			if (!activeMaterial->hasProgram()) {
				continue;
			}

			const gl::Program* const rimProgram{ activeMaterial->getProgram().get() };

			if (activeProgram != rimProgram)
			{
				activeProgram = rimProgram;
				activeProgram->use();
			}

			GL_CHECK_ERROR(glDrawArraysInstanced(GL_PATCHES, 0, 4, 2 * subDivision.x + 2 * subDivision.y));
		}
	}
}

}

#pragma once

#include <dunes/core/water.hpp>
#include <sthe/core/pipeline.hpp>
#include <sthe/core/scene.hpp>
#include <sthe/core/material.hpp>
#include <sthe/components/camera.hpp>
#include <sthe/gl/framebuffer.hpp>
#include <sthe/gl/program.hpp>
#include <sthe/gl/buffer.hpp>
#include <memory>
#include <array>
#include <vector>

namespace dunes
{
using namespace sthe;
namespace uniform
{

struct DunesPipeline
{
	float time;
	float deltaTime;
	glm::ivec2 resolution;
	glm::mat4 projectionMatrix;
	glm::mat4 viewMatrix;
	glm::mat4 inverseViewMatrix;
	glm::mat4 viewProjectionMatrix;
	sthe::uniform::Environment environment;

	glm::mat4 modelMatrix;
	glm::mat4 inverseModelMatrix;
	glm::mat4 modelViewMatrix;
	glm::mat4 inverseModelViewMatrix;
	glm::ivec4 userID;

	union
	{
		sthe::uniform::Material material;
		sthe::uniform::Terrain terrain;
		dunes::uniform::Water water;
	};
};

}

class DunesPipeline : public Pipeline
{
public:
	// Constructors
	DunesPipeline();
	DunesPipeline(const DunesPipeline& t_dunesPipeline) = delete;
	DunesPipeline(DunesPipeline&& t_dunesPipeline) = default;

	// Destructor
	~DunesPipeline() = default;

	// Operators
	DunesPipeline& operator=(const DunesPipeline& t_dunesPipeline) = delete;
	DunesPipeline& operator=(DunesPipeline&& t_dunesPipeline) = default;

	// Functionality
	void use() override;
	void disuse() override;
	void render(const Scene& t_scene, const Camera& t_camera) override;
private:
	// Functionality
	void setup(const Scene& t_scene, const Camera& t_camera);
	void meshRendererZPass(const Scene& t_scene);
	void meshRendererPass(const Scene& t_scene);
	void terrainRendererPass(const Scene& t_scene);
	void waterRendererPass(const Scene& t_scene);
	void setupFramebuffers(int width, int height);

	// Attributes
	uniform::DunesPipeline m_data;
	std::shared_ptr<gl::VertexArray> m_vertexArray;
	std::shared_ptr<gl::Program> m_meshProgram;
	std::shared_ptr<gl::Program> m_terrainProgram;
	std::shared_ptr<gl::Buffer> m_pipelineBuffer;
	std::shared_ptr<gl::Framebuffer> m_terrainFrameBuffer;
	std::shared_ptr<gl::Framebuffer> m_waterFrameBuffer;
	std::shared_ptr<sthe::gl::Texture2D> m_terrainDiffuseMap;
	std::shared_ptr<sthe::gl::Texture2D> m_terrainPositionMap;
	std::shared_ptr<sthe::gl::Texture2D> m_depthMap;
	std::shared_ptr<sthe::CustomMaterial> m_screenMaterial;
	std::shared_ptr<sthe::gl::Program> m_screenProgram;
	std::shared_ptr<sthe::gl::Program> m_zPassProgram;

	std::shared_ptr<sthe::gl::Texture2D> m_waterDiffuseMap;
	std::shared_ptr<sthe::gl::Texture2D> m_waterPositionOffsetMap;
};

}

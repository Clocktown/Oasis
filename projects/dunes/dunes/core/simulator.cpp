#include "simulator.hpp"
#include "simulation_parameters.hpp"
#include "render_parameters.hpp"
#include "launch_parameters.hpp"
#include <dunes/device/constants.cuh>
#include <dunes/device/kernels.cuh>
#include <dunes/util/io.hpp>
#include <sthe/sthe.hpp>
#include <cufft.h>
#include <vector>

#include <random>

namespace dunes
{
	// Constructor
	Simulator::Simulator() :
		m_timeScale{ 1.0f },
		m_fixedDeltaTime{ 0.02f },
		m_terrain{ std::make_shared<sthe::Terrain>() },
		m_material{ std::make_shared<sthe::CustomMaterial>() },
		m_program{ std::make_shared<sthe::gl::Program>() },
		m_rimMaterial{ std::make_shared<sthe::CustomMaterial>() },
		m_rimProgram{ std::make_shared<sthe::gl::Program>() },
		m_water{ std::make_shared<Water>() },
		m_waterMaterial{ std::make_shared<sthe::CustomMaterial>() },
		m_waterProgram{ std::make_shared<sthe::gl::Program>() },
		m_waterRimMaterial{ std::make_shared<sthe::CustomMaterial>() },
		m_waterRimProgram{ std::make_shared<sthe::gl::Program>() },
		m_terrainMap{ std::make_shared<sthe::gl::Texture2D>() },
		m_windMap{ std::make_shared<sthe::gl::Texture2D>() },
		m_resistanceMap{ std::make_shared<sthe::gl::Texture2D>() },
		m_waterVelocityMap{ std::make_shared<sthe::gl::Texture2D>() },
		m_sedimentMap{ std::make_shared<sthe::gl::Texture2D>() },
		m_fluxMap{ std::make_shared<sthe::gl::Texture2D>() },
		m_terrainMoistureMap{ std::make_shared<sthe::gl::Texture2D>() },
		m_shadowMap{ std::make_shared<sthe::gl::Texture2D>() },
		m_vegetationHeightMap{ std::make_shared<sthe::gl::Texture2D>() },
		m_textureDescriptor{},
		m_isAwake{ false },
		m_isPaused{ false },
		m_reinitializeWindWarping{ false },
		m_uploadVegTypes{ false },
		m_renderParameterBuffer{ std::make_shared<sthe::gl::Buffer>(static_cast<int>(sizeof(RenderParameters)), 1) },
		m_coverageMap{ nullptr },
		m_coverage{ std::numeric_limits<float>::quiet_NaN() },
		m_coverageThreshold{ 0.001f }
	{
		const int device{ 0 };
		int smCount;
		int smThreadCount;
		CU_CHECK_ERROR(cudaSetDevice(device));
		CU_CHECK_ERROR(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
		CU_CHECK_ERROR(cudaDeviceGetAttribute(&smThreadCount, cudaDevAttrMaxThreadsPerMultiProcessor, device));
		const float threadCount{ static_cast<float>(smCount * smThreadCount) };
		
		m_launchParameters.blockSize1D = 512;
		m_launchParameters.blockSize2D = dim3{ 8, 8 };
		
		// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
		m_launchParameters.optimalBlockSize1D = 256;
		m_launchParameters.optimalBlockSize2D = dim3{ 16, 16 };
		m_launchParameters.optimalGridSize1D = static_cast<unsigned int>(threadCount / static_cast<float>(m_launchParameters.optimalBlockSize1D));
		m_launchParameters.optimalGridSize2D.x = nextPowerOfTwo(static_cast<unsigned int>(glm::sqrt(threadCount / static_cast<float>(m_launchParameters.optimalBlockSize2D.x * m_launchParameters.optimalBlockSize2D.y))));
		m_launchParameters.optimalGridSize2D.y = m_launchParameters.optimalGridSize2D.x;
		m_launchParameters.optimalGridSize2D.z = 1;

		m_terrain->setHeightMap(m_terrainMap);
		m_terrain->addLayer(std::make_shared<sthe::TerrainLayer>(glm::vec3(194.0f, 178.0f, 128.0f) / 255.0f));

		m_water->setHeightMap(m_terrainMap);

		m_program->setPatchVertexCount(4);
		m_program->attachShader(sthe::gl::Shader{ GL_VERTEX_SHADER, sthe::getShaderPath() + "terrain/phong.vert" });
		m_program->attachShader(sthe::gl::Shader{ GL_TESS_CONTROL_SHADER, sthe::getShaderPath() + "terrain/phong.tesc" });
		m_program->attachShader(sthe::gl::Shader{ GL_TESS_EVALUATION_SHADER, getShaderPath() + "terrain/phong.tese" });
		m_program->attachShader(sthe::gl::Shader{ GL_FRAGMENT_SHADER, getShaderPath() + "terrain/phong.frag" });
		m_program->link();

		m_rimProgram->setPatchVertexCount(4);
		m_rimProgram->attachShader(sthe::gl::Shader{ GL_VERTEX_SHADER, getShaderPath() + "terrainRim/phong.vert" });
		m_rimProgram->attachShader(sthe::gl::Shader{ GL_TESS_CONTROL_SHADER, getShaderPath() + "terrainRim/phong.tesc" });
		m_rimProgram->attachShader(sthe::gl::Shader{ GL_TESS_EVALUATION_SHADER, getShaderPath() + "terrainRim/phong.tese" });
		m_rimProgram->attachShader(sthe::gl::Shader{ GL_FRAGMENT_SHADER, getShaderPath() + "terrainRim/phong.frag" });
		m_rimProgram->link();

		m_waterProgram->setPatchVertexCount(4);
		m_waterProgram->attachShader(sthe::gl::Shader{ GL_VERTEX_SHADER, getShaderPath() + "water/phong.vert" });
		m_waterProgram->attachShader(sthe::gl::Shader{ GL_TESS_CONTROL_SHADER, getShaderPath() + "water/phong.tesc" });
		m_waterProgram->attachShader(sthe::gl::Shader{ GL_TESS_EVALUATION_SHADER, getShaderPath() + "water/phong.tese" });
		m_waterProgram->attachShader(sthe::gl::Shader{ GL_FRAGMENT_SHADER, getShaderPath() + "water/phong.frag" });
		m_waterProgram->link();

		m_waterRimProgram->setPatchVertexCount(4);
		m_waterRimProgram->attachShader(sthe::gl::Shader{ GL_VERTEX_SHADER, getShaderPath() + "waterRim/phong.vert" });
		m_waterRimProgram->attachShader(sthe::gl::Shader{ GL_TESS_CONTROL_SHADER, getShaderPath() + "waterRim/phong.tesc" });
		m_waterRimProgram->attachShader(sthe::gl::Shader{ GL_TESS_EVALUATION_SHADER, getShaderPath() + "waterRim/phong.tese" });
		m_waterRimProgram->attachShader(sthe::gl::Shader{ GL_FRAGMENT_SHADER, getShaderPath() + "water/phong.frag" });
		m_waterRimProgram->link();

		m_vegPrefabs.program = std::make_shared<sthe::gl::Program>();
		m_vegPrefabs.program->attachShader(sthe::gl::Shader{ GL_VERTEX_SHADER, getShaderPath() + "vegetation/phong.vert" });
		m_vegPrefabs.program->attachShader(sthe::gl::Shader{ GL_FRAGMENT_SHADER, getShaderPath() + "vegetation/phong.frag" });
		m_vegPrefabs.program->link();

		m_material->setProgram(m_program);
		m_material->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0, m_windMap);
		m_material->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0 + 1, m_resistanceMap);
		m_material->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0 + 2, m_terrainMoistureMap);
		m_material->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0 + 3, m_sedimentMap);

		m_rimMaterial->setProgram(m_rimProgram);
		m_rimMaterial->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0, m_windMap);
		m_rimMaterial->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0 + 1, m_resistanceMap);
		m_rimMaterial->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0 + 2, m_terrainMoistureMap);
		m_rimMaterial->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0 + 3, m_sedimentMap);

		m_waterMaterial->setProgram(m_waterProgram);
		m_waterMaterial->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0, m_windMap);
		m_waterMaterial->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0 + 1, m_resistanceMap);
		m_waterMaterial->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0 + 2, m_terrainMoistureMap);
		m_waterMaterial->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0 + 3, m_sedimentMap);
		m_waterMaterial->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0 + 4, m_shadowMap);
		m_waterMaterial->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0 + 5, m_vegetationHeightMap);

		m_waterRimMaterial->setProgram(m_waterRimProgram);
		m_waterRimMaterial->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0, m_windMap);
		m_waterRimMaterial->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0 + 1, m_resistanceMap);
		m_waterRimMaterial->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0 + 2, m_terrainMoistureMap);
		m_waterRimMaterial->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0 + 3, m_sedimentMap);
		m_waterRimMaterial->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0 + 4, m_shadowMap);
		m_waterRimMaterial->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0 + 5, m_vegetationHeightMap);

		m_textureDescriptor.addressMode[0] = cudaAddressModeWrap;
		m_textureDescriptor.addressMode[1] = cudaAddressModeWrap;
		m_textureDescriptor.filterMode = cudaFilterModeLinear;
		m_textureDescriptor.normalizedCoords = 0;

		m_renderParameterBuffer->bind(GL_UNIFORM_BUFFER, STHE_UNIFORM_BUFFER_CUSTOM0);
		m_renderParameterBuffer->upload(reinterpret_cast<char*>(&m_renderParameters), sizeof(RenderParameters));

		m_watches.resize(20);
		m_watchTimings.resize(20);
		m_meanWatchTimings.resize(20);
		resetPerformanceAverages();

		m_vegPrefabs.files = { dunes::getResourcePath() + "models\\MapleFall.obj",
			                   dunes::getResourcePath() + "models\\BushFlowerSmall.obj",
			                   dunes::getResourcePath() + "models\\seaweed.obj" };

		std::array<VegetationType, 3> vegTypes
		{
			VegetationType
			{
				20.0f,
				0.1f,
				0.0001f,
				0.02f,
				0.0f,
				100.f,
				0.2f,
				{ 1.f, 1.f },
				0.1f,
				1.f,
				0.1f,
				0.2f,
				0.8f,
				1.f,
				0.2f,
				{ 0.75f, 0.25f },
				0.5f,
				0.1f,
				10.f,
				100.f,
				0.01f,
				{0.3f, 1.7f}
			},
			VegetationType
			{
				2.f,
				0.2f,
				0.01f,
				0.1f,
				0.1f,
				20.f,
				0.2f,
				{2.f, 0.5f},
				0.05f,
				2.f,
				0.3f,
				0.2f,
				0.9f,
				1.f,
				0.4f,
				{0.75f, 0.75f},
				0.75f,
				1.f,
				1000000.f,
				10.f,
				0.01f,
				{0.0f, 0.75f}
		},
		VegetationType
		{
				3.f,
				0.3f,
				0.01f,
				0.1f,
				1.f,
				30.f,
				0.1f,
				{0.5f, 1.f},
				0.1f,
				1.f,
				1.0f,
				0.f,
				1.f,
				1.f,
				1.0f,
				{0.75f, 0.25f},
				0.25f,
				0.5f,
				1.f,
				0.1f,
				0.001f,
				{0.1f, 0.4f}
			}
		};

		for (int i = 0; i < 3; ++i) {
			setVegetationType(i, vegTypes[i]);
		}

		m_vegMatrix.resize(c_maxVegTypeCount * c_maxVegTypeCount, 1.0f);
		m_vegMatrix[1 + 0 * c_maxVegTypeCount] = 0.5f;
		m_vegMatrix[0 + 1 * c_maxVegTypeCount] = 2.0f;
	}

	void Simulator::resetPerformanceAverages() {
		m_perfStep = 0;
		std::fill(m_watchTimings.begin(), m_watchTimings.end(), 0.f);
		std::fill(m_meanWatchTimings.begin(), m_meanWatchTimings.end(), 0.f);
	}

	// Destructor
	Simulator::~Simulator()
	{
		CUFFT_CHECK_ERROR(cufftDestroy(m_launchParameters.fftPlan));
		CUFFT_CHECK_ERROR(cufftDestroy(m_launchParameters.projection.planR2C));
		CUFFT_CHECK_ERROR(cufftDestroy(m_launchParameters.projection.planC2R));
	}

	// Functionality
	void Simulator::reinitialize(const glm::ivec2& t_gridSize, const float t_gridScale)
	{
		m_time = 0.f;
		m_timeStep = 0;
		resetPerformanceAverages();
		//STHE_ASSERT(t_gridSize.x > 0 && (t_gridSize.x & (t_gridSize.x - 1)) == 0, "Grid size x must be a power of 2");
		//STHE_ASSERT(t_gridSize.y > 0 && (t_gridSize.y & (t_gridSize.y - 1)) == 0, "Grid size y must be a power of 2");
		STHE_ASSERT(t_gridScale != 0.0f, "Grid scale cannot be 0");

		m_simulationParameters.gridSize.x = t_gridSize.x;
		m_simulationParameters.gridSize.y = t_gridSize.y;
		m_simulationParameters.cellCount = t_gridSize.x * t_gridSize.y;
		m_simulationParameters.gridScale = t_gridScale;
		m_simulationParameters.rGridScale = 1.0f / t_gridScale;

		const float uniformGridScale = 2.f * c_maxVegetationRadius;
		const glm::vec2 gridDim = glm::vec2(t_gridSize) * t_gridScale;
		const glm::ivec2 uniformGridSize = glm::ivec2(glm::ceil(gridDim / uniformGridScale)); // 20.f max radius
		m_simulationParameters.maxVegCount = m_launchParameters.maxVegCount;

		if (m_launchParameters.fftPlan != 0)
		{
			CUFFT_CHECK_ERROR(cufftDestroy(m_launchParameters.fftPlan));
		}

		//CUFFT_CHECK_ERROR(cufftPlan2d(&m_launchParameters.fftPlan, m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y, cufftType::CUFFT_C2C));

		if (m_isAwake)
		{
			awake();
		}
		else
		{
			getGameObject().getTransform().setLocalPosition(-0.5f * getGameObject().getTransform().getLocalScale() * glm::vec3{ gridDim.x, 0.0f, gridDim.y });
		}
	}

	void Simulator::awake()
	{
		getGameObject().getTransform().setLocalPosition(-0.5f * getGameObject().getTransform().getLocalScale() * m_simulationParameters.gridScale * glm::vec3{ m_simulationParameters.gridSize.x, 0.0f, m_simulationParameters.gridSize.y });

		setupLaunchParameters();
		setupTerrain();
		setupArrays();
		setupBuffers();
		setupWindWarping();
		setupProjection();
		setupVegPrefabs();
		setupAdaptiveGrid();

		map();

		initializeTerrain(m_launchParameters, m_initializationParameters);
		initializeWindWarping(m_launchParameters, m_simulationParameters);
		initializeVegetation(m_launchParameters);
		// TODO: slope Buffer not properly initialized when this runs
		getVegetationCount(m_launchParameters, m_simulationParameters);
		vegetation(m_launchParameters, m_simulationParameters, m_watches);
		venturi(m_launchParameters);
		windWarping(m_launchParameters);
		pressureProjection(m_launchParameters, m_simulationParameters);
		windShadow(m_launchParameters);

		unmap();

		m_isAwake = true;
	}

	void Simulator::updateWindShadow() {
		map();
		windShadow(m_launchParameters);
		unmap();
	}

	bool Simulator::queryTimeStepHappened() {
		return m_timestepHappened;
	}

	void Simulator::update()
	{
		m_timestepHappened = false;
		if (!m_isPaused)
		{
			m_timestepHappened = true;
			if (m_enableBidirectional) {
				float time = m_time / m_windBidirectionalBaseTime;
				time = fmod(time, m_bidirectionalStrengthBased ? 2.f : (m_windBidirectionalR + 1.f));
				const float angle = time >= 1.f ? m_firstWindAngle : m_secondWindAngle;
				setWindDirection(angle);

				if (m_bidirectionalStrengthBased) {
					const float speed = time >= 1.f ? m_windSpeed : m_windSpeed / m_windBidirectionalR;
					applyWindSpeed(speed);
				}
			}

			m_simulationParameters.timestep = m_timeStep;
			map();

			if (m_reinitializeWindWarping)
			{
				initializeWindWarping(m_launchParameters, m_simulationParameters);
				m_reinitializeWindWarping = false;
			}

			if (m_uploadVegTypes) {
				m_vegTypeBuffer.upload(&m_vegTypes, 1);
				m_uploadVegTypes = false;
			}

			if (m_constantCoverage && ((m_timeStep % m_spawnSteps) == 0) && (m_coverage < m_targetCoverage)) {
				addSandForCoverage(m_launchParameters, m_simulationParameters.gridSize, m_coverageSpawnUniform, m_coverageRadius, m_coverageSpawnAmount);
			}
			else if (m_constantCoverageAllowRemove && m_constantCoverage && ((m_timeStep % m_spawnSteps) == 0) && (m_coverage > m_targetCoverage)) {
				addSandForCoverage(m_launchParameters, m_simulationParameters.gridSize, true, m_coverageRadius, -m_coverageSubtractAmount);
			}

			m_watches[0].start();
			m_watches[1].start();
			vegetation(m_launchParameters, m_simulationParameters, m_watches);
			m_watches[1].stop();
			m_watches[6].start();
			m_watches[7].start();
			venturi(m_launchParameters);
			m_watches[7].stop();
			m_watches[8].start();
			windWarping(m_launchParameters);
			m_watches[8].stop();
			m_watches[9].start();
			pressureProjection(m_launchParameters, m_simulationParameters);
			m_watches[9].stop();
			m_watches[10].start();
			windShadow(m_launchParameters);
			m_watches[10].stop();
			m_watches[11].start();
			continuousSaltation(m_launchParameters, m_simulationParameters);
			m_watches[11].stop();
			m_watches[12].start();
			continuousReptation(m_launchParameters, m_simulationParameters);
			m_watches[12].stop();
			m_watches[13].start();
			avalanching(m_launchParameters, m_simulationParameters);
			m_watches[13].stop();
			m_watches[14].start();
			bedrockAvalanching(m_launchParameters);
			m_watches[14].stop();
			m_watches[6].stop();
			m_watches[15].start();
			m_watches[16].start();
			rain(m_launchParameters);
			m_watches[16].stop();
			m_watches[17].start();
			transport(m_launchParameters, m_simulationParameters);
			m_watches[17].stop();
			m_watches[18].start();
			moisture(m_launchParameters, m_simulationParameters);
			m_watches[18].stop();
			m_watches[19].start();
			sediment(m_launchParameters, m_simulationParameters);
			m_watches[19].stop();
			m_watches[15].stop();
			m_watches[0].stop();

			if (m_coverageMap) {
				calculateCoverage();
			}

			m_time += m_simulationParameters.deltaTime;
			m_timeStep++;
			m_perfStep++;

			unmap();

			int userID{ 0 };

			for (int i{ 0 }; i < c_maxVegTypeCount; ++i)
			{
				for (sthe::MeshRenderer* const meshRenderer : m_vegPrefabs.meshRenderers[i])
				{
					meshRenderer->setInstanceCount(m_launchParameters.vegCountsPerType[i]);
					meshRenderer->setUserID({ userID, 0, 0, 0 });
				}

				userID += m_launchParameters.vegCountsPerType[i];
			}

			for (int i = 0; i < m_watches.size(); ++i) {
				float t = m_watches[i].getTime();
				m_watchTimings[i] = t;
				m_meanWatchTimings[i] = ((m_meanWatchTimings[i] * (m_perfStep - 1)) + t) / m_perfStep;
			}

			if (m_stopIterations != 0 && m_perfStep % m_stopIterations == 0) {
				m_isPaused = true;
			}
		}
	}

	const std::vector<float>& Simulator::getWatchTimings() {
		return m_watchTimings;
	}
	const std::vector<float>& Simulator::getMeanWatchTimings() {
		return m_meanWatchTimings;
	}

	void Simulator::resume()
	{
		m_isPaused = false;
	}

	void Simulator::pause()
	{
		m_isPaused = true;
	}

	void Simulator::setupLaunchParameters()
	{
		m_launchParameters.gridSize1D = static_cast<unsigned int>(glm::ceil(static_cast<float>(m_simulationParameters.cellCount) / static_cast<float>(m_launchParameters.blockSize1D)));
		m_launchParameters.gridSize2D.x = static_cast<unsigned int>(glm::ceil(static_cast<float>(m_simulationParameters.gridSize.x) / static_cast<float>(m_launchParameters.blockSize2D.x)));
		m_launchParameters.gridSize2D.y = static_cast<unsigned int>(glm::ceil(static_cast<float>(m_simulationParameters.gridSize.y) / static_cast<float>(m_launchParameters.blockSize2D.y)));
	}

	void Simulator::setupTerrain()
	{
		m_terrainRenderer = &getGameObject().addComponent<sthe::TerrainRenderer>();
		m_terrainRenderer->setTerrain(m_terrain);
		m_terrainRenderer->setMaterial(m_material);
		m_terrainRenderer->setRimMaterial(m_rimMaterial);

		m_waterRenderer = &getGameObject().addComponent<WaterRenderer>();
		m_waterRenderer->setWater(m_water);
		m_waterRenderer->setMaterial(m_waterMaterial);
		m_waterRenderer->setRimMaterial(m_waterRimMaterial);

		m_terrain->setGridSize(glm::ivec2{ m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y });
		m_terrain->setGridScale(m_simulationParameters.gridScale);

		m_water->setGridSize(glm::ivec2{ m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y });
		m_water->setGridScale(m_simulationParameters.gridScale);

		m_terrainMap->reinitialize(m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y, GL_RGBA32F, false);
		m_windMap->reinitialize(m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y, GL_RG32F, false);
		m_resistanceMap->reinitialize(m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y, GL_RGBA32F, false);
		m_waterVelocityMap->reinitialize(m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y, GL_RG32F, false);
		m_fluxMap->reinitialize(m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y, GL_RGBA32F, false);
		m_sedimentMap->reinitialize(m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y, GL_R32F, false);
		m_terrainMoistureMap->reinitialize(m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y, GL_R32F, false);
		m_shadowMap->reinitialize(m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y, GL_RG32F, false);
		m_vegetationHeightMap->reinitialize(m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y, GL_RG32F, false);
	}

	void Simulator::setupArrays()
	{
		m_terrainArray.reinitialize(*m_terrainMap);
		m_windArray.reinitialize(*m_windMap);
		m_resistanceArray.reinitialize(*m_resistanceMap);
		m_waterVelocityArray.reinitialize(*m_waterVelocityMap);
		m_fluxArray.reinitialize(*m_fluxMap);
		m_sedimentArray.reinitialize(*m_sedimentMap);
		m_terrainMoistureArray.reinitialize(*m_terrainMoistureMap);
		m_vegetationHeightArray.reinitialize(*m_vegetationHeightMap);
		m_shadowArray.reinitialize(*m_shadowMap);
	}

	void Simulator::setMaxVegCount(float c) {
		m_launchParameters.maxVegCount = c;
	}

	void Simulator::setupBuffers()
	{
		m_slabBuffer.reinitialize(m_simulationParameters.cellCount, sizeof(float));
		m_simulationParameters.slabBuffer = m_slabBuffer.getData<float>();

		m_tmpBuffer.reinitialize(5 * m_simulationParameters.cellCount, sizeof(float));
		m_launchParameters.tmpBuffer = m_tmpBuffer.getData<float>();

		const int maxCount = max(1000000, m_launchParameters.maxVegCount);
		const int counts[1 + c_maxVegTypeCount]{0.f};
		m_launchParameters.vegCount = counts[0];
		//m_launchParameters.maxVegCount = maxCount;
		m_launchParameters.vegetationGridSize1D = counts[0] == 0 ? 1 : static_cast<unsigned int>(glm::ceil(static_cast<float>(counts[0]) / static_cast<float>(m_launchParameters.blockSize1D)));

		m_vegPrefabs.buffer = std::make_shared<sthe::gl::Buffer>(maxCount, sizeof(Vegetation));
		m_vegPrefabs.mapBuffer = std::make_shared<sthe::gl::Buffer>(maxCount, sizeof(int));
	
		m_vegBuffer.reinitialize(*m_vegPrefabs.buffer);
		m_vegMapBuffer.reinitialize(*m_vegPrefabs.mapBuffer);

		m_vegCountBuffer.reinitialize(1 + c_maxVegTypeCount, sizeof(int));
		m_vegCountBuffer.upload(counts, 1 + c_maxVegTypeCount);
		m_simulationParameters.vegCountBuffer = m_vegCountBuffer.getData<int>();

		m_vegTypeBuffer.reinitialize(1, sizeof(VegetationTypeSoA));
		m_vegTypeBuffer.upload(&m_vegTypes, 1);
		m_simulationParameters.vegTypeBuffer = m_vegTypeBuffer.getData<VegetationTypeSoA>();

		m_vegMatrixBuffer.reinitialize(c_maxVegTypeCount * c_maxVegTypeCount, sizeof(float));
		m_vegMatrixBuffer.upload(m_vegMatrix);
		m_simulationParameters.vegMatrixBuffer = m_vegMatrixBuffer.getData<float>();

		std::random_device rd;
		std::mt19937 gen(rd());

		std::vector<uint4> seeds(max(maxCount, m_simulationParameters.cellCount));
		for (uint4& s : seeds) {
			s.x = gen();
			s.y = gen();
			s.z = gen();
			s.w = gen();
		}

		m_seedBuffer.reinitialize(seeds);
		m_simulationParameters.seedBuffer = m_seedBuffer.getData<uint4>();
	}

	void Simulator::setUseBilinear(const bool t_useBilinear) {
		m_launchParameters.useBilinear = t_useBilinear;
	}

	void Simulator::setupWindWarping()
	{
		const int2 gridSize{ m_simulationParameters.gridSize };
		CUFFT_CHECK_ERROR(cufftPlan2d(&m_launchParameters.fftPlan, gridSize.y, gridSize.x, cufftType::CUFFT_C2C));

		for (int i{ 0 }; i < 4; ++i)
		{
			sthe::cu::Buffer& buffer{ m_windWarpingBuffers[i] };
			buffer.reinitialize(2 * m_simulationParameters.cellCount, sizeof(cuComplex));

			m_launchParameters.windWarping.gaussKernels[i] = buffer.getData<cuComplex>();
			m_launchParameters.windWarping.smoothedHeights[i] = m_launchParameters.windWarping.gaussKernels[i] + m_simulationParameters.cellCount;
		}
	}

	void Simulator::setupProjection()
	{
		const int2 gridSize{ m_simulationParameters.gridSize };
		const int cellCount{ m_simulationParameters.cellCount };

		CUFFT_CHECK_ERROR(cufftPlan2d(&m_launchParameters.projection.planR2C, gridSize.y, gridSize.x, cufftType::CUFFT_R2C));
		CUFFT_CHECK_ERROR(cufftPlan2d(&m_launchParameters.projection.planC2R, gridSize.y, gridSize.x, cufftType::CUFFT_C2R));

		const int size{ (gridSize.x / 2 + 1) * gridSize.y };

		m_velocityBuffer.reinitialize(4 * size, sizeof(float));
		m_launchParameters.projection.velocities[0] = m_velocityBuffer.getData<float>();
		m_launchParameters.projection.velocities[1] = m_launchParameters.projection.velocities[0] + 2 * size;
	}

	void Simulator::setupVegPrefabs()
	{
		for (int i = 0; i < m_simulationParameters.vegTypeCount; ++i) {
			if (m_vegPrefabs.gameObjects[i]) {
				getScene().removeGameObject(*m_vegPrefabs.gameObjects[i]);
			}
		}

		for (int i{ 0 }; i < m_simulationParameters.vegTypeCount; ++i)
		{
			sthe::Importer importer{ m_vegPrefabs.files[i].string() };

			sthe::GameObject& gameObject{ importer.importModel(getScene(), m_vegPrefabs.program) };
			gameObject.getTransform().setParent(&getGameObject().getTransform(), false);

			m_vegPrefabs.gameObjects[i] = &gameObject;
			m_vegPrefabs.meshRenderers[i] = gameObject.getComponentsInChildren<sthe::MeshRenderer>();

			for (sthe::MeshRenderer* const meshRenderer : m_vegPrefabs.meshRenderers[i])
			{
				meshRenderer->setInstanceCount(0);
			}

			for (auto& material : importer.getMaterials())
			{
				material->setBuffer(GL_SHADER_STORAGE_BUFFER, STHE_STORAGE_BUFFER_CUSTOM0, m_vegPrefabs.buffer);
				material->setBuffer(GL_SHADER_STORAGE_BUFFER, STHE_STORAGE_BUFFER_CUSTOM0 + 1, m_vegPrefabs.mapBuffer);
			}
		}
	}

	void Simulator::setupAdaptiveGrid()
	{
		int cellCount{ 0 };

		for (int i{ 0 }; i < m_simulationParameters.adaptiveGrid.layerCount; ++i)
		{
			m_simulationParameters.adaptiveGrid.gridSizes[i] = make_int2(ceilf(make_float2(m_simulationParameters.gridSize) * m_simulationParameters.gridScale / m_simulationParameters.adaptiveGrid.gridScales[i]));
			m_simulationParameters.adaptiveGrid.cellCounts[i] = m_simulationParameters.adaptiveGrid.gridSizes[i].x * m_simulationParameters.adaptiveGrid.gridSizes[i].y;
			
			cellCount += m_simulationParameters.adaptiveGrid.cellCounts[i];
		}

		m_adaptiveGrid.gridBuffer.reinitialize(cellCount + int(m_simulationParameters.adaptiveGrid.layerCount), sizeof(unsigned int));
		cellCount = 0;

		for (int i{ 0 }; i < m_simulationParameters.adaptiveGrid.layerCount; ++i)
		{
			m_simulationParameters.adaptiveGrid.gridBuffer[i] = m_adaptiveGrid.gridBuffer.getData<unsigned int>() + cellCount + i;
			cellCount += m_simulationParameters.adaptiveGrid.cellCounts[i];
		}

		m_adaptiveGrid.keyBuffer.reinitialize(m_simulationParameters.maxVegCount, sizeof(unsigned int));
		m_simulationParameters.adaptiveGrid.keyBuffer = m_adaptiveGrid.keyBuffer.getData<unsigned int>();
		m_adaptiveGrid.indexBuffer.reinitialize(m_simulationParameters.maxVegCount, sizeof(unsigned int));
		m_simulationParameters.adaptiveGrid.indexBuffer = m_adaptiveGrid.indexBuffer.getData<unsigned int>();
		m_adaptiveGrid.vegBuffer.reinitialize(m_simulationParameters.maxVegCount, sizeof(Vegetation));
		m_simulationParameters.adaptiveGrid.vegBuffer = m_adaptiveGrid.vegBuffer.getData<Vegetation>();
	}

	void Simulator::setupCoverageCalculation() {
		m_coverageMap = std::make_unique<sthe::cu::Buffer>(int(m_simulationParameters.cellCount), int(sizeof(unsigned int)));
		m_coverage = 0.f;
	}

	void Simulator::calculateCoverage() {
		m_coverage = coverage(m_launchParameters, m_coverageMap->getData<unsigned int>(), m_simulationParameters.cellCount, m_coverageThreshold);
	}

	void Simulator::cleanupCoverageCalculation() {
		m_coverageMap = nullptr;
		m_coverage = std::numeric_limits<float>::quiet_NaN();
	}

	float Simulator::getCoverage() {
		return m_coverage;
	}

	void Simulator::setTargetCoverage(const float t_targetCoverage)
	{
		m_targetCoverage = t_targetCoverage;
	}

	void Simulator::setCoverageSpawnAmount(const float t_amount)
	{
		m_coverageSpawnAmount = t_amount;
	}

	void Simulator::setCoverageSubtractAmount(const float t_amount)
	{
		m_coverageSubtractAmount = t_amount;
	}

	void Simulator::setCoverageSpawnUniform(const bool t_uniform) {
		m_coverageSpawnUniform = t_uniform;
	}
	void Simulator::setCoverageRadius(const int t_radius) {
		m_coverageRadius = t_radius;
	}

	void Simulator::setSpawnSteps(const int t_steps)
	{
		m_spawnSteps = t_steps;
	}

	void Simulator::setConstantCoverage(const bool t_constantCoverage)
	{
		m_constantCoverage = t_constantCoverage;
	}

	void Simulator::setConstantCoverageAllowRemove(const bool t_constantCoverageAllowRemove)
	{
		m_constantCoverageAllowRemove = t_constantCoverageAllowRemove;
	}

	void Simulator::setProjectionMode(const ProjectionMode t_mode)
	{
		m_launchParameters.projection.mode = t_mode;
	}

	void Simulator::map()
	{
		m_simulationParameters.deltaTime = m_launchParameters.timeMode == TimeMode::DeltaTime ? sthe::getApplication().getDeltaTime() : m_fixedDeltaTime;
		m_simulationParameters.deltaTime *= m_timeScale;

		m_terrainArray.map();
		m_simulationParameters.terrainArray.surface = m_terrainArray.recreateSurface();
		m_simulationParameters.terrainArray.texture = m_terrainArray.recreateTexture(m_textureDescriptor);

		m_windArray.map();
		m_simulationParameters.windArray.surface = m_windArray.recreateSurface();
		m_simulationParameters.windArray.texture = m_windArray.recreateTexture(m_textureDescriptor);

		m_resistanceArray.map();
		m_simulationParameters.resistanceArray.surface = m_resistanceArray.recreateSurface();
		m_simulationParameters.resistanceArray.texture = m_resistanceArray.recreateTexture(m_textureDescriptor);

		m_fluxArray.map();
		m_simulationParameters.fluxArray.surface = m_fluxArray.recreateSurface();
		m_simulationParameters.fluxArray.texture = m_fluxArray.recreateTexture(m_textureDescriptor);

		m_waterVelocityArray.map();
		m_simulationParameters.velocityArray.surface = m_waterVelocityArray.recreateSurface();
		m_simulationParameters.velocityArray.texture = m_waterVelocityArray.recreateTexture(m_textureDescriptor);

		m_sedimentArray.map();
		m_simulationParameters.sedimentArray.surface = m_sedimentArray.recreateSurface();
		m_simulationParameters.sedimentArray.texture = m_sedimentArray.recreateTexture(m_textureDescriptor);

		m_terrainMoistureArray.map();
		m_simulationParameters.moistureArray.surface = m_terrainMoistureArray.recreateSurface();
		m_simulationParameters.moistureArray.texture = m_terrainMoistureArray.recreateTexture(m_textureDescriptor);

		m_shadowArray.map();
		m_simulationParameters.shadowArray.surface = m_shadowArray.recreateSurface();
		m_simulationParameters.shadowArray.texture = m_shadowArray.recreateTexture(m_textureDescriptor);

		m_vegetationHeightArray.map();
		m_simulationParameters.vegHeightArray.surface = m_vegetationHeightArray.recreateSurface();
		m_simulationParameters.vegHeightArray.texture = m_vegetationHeightArray.recreateTexture(m_textureDescriptor);

		m_vegBuffer.map(sizeof(Vegetation));
		m_simulationParameters.vegBuffer = m_vegBuffer.getData<Vegetation>();

		m_vegMapBuffer.map(sizeof(int));
		m_simulationParameters.vegMapBuffer = m_vegMapBuffer.getData<int>();

		upload(m_simulationParameters);
	}

	void Simulator::unmap()
	{
		m_terrainArray.unmap();
		m_windArray.unmap();
		m_resistanceArray.unmap();
		m_fluxArray.unmap();
		m_waterVelocityArray.unmap();
		m_sedimentArray.unmap();
		m_terrainMoistureArray.unmap();
		m_shadowArray.unmap();
		m_vegetationHeightArray.unmap();
		m_vegBuffer.unmap();
		m_vegMapBuffer.unmap();
	}

	// Setters
	void Simulator::setSecondWindAngle(const float t_windAngle)
	{
		m_secondWindAngle = t_windAngle;
	}

	void Simulator::enableBidirectional(const bool t_enable)
	{
		m_enableBidirectional = t_enable;
	}

	void Simulator::setBidirectionalStrengthBased(const bool t_sBased) {
		m_bidirectionalStrengthBased = t_sBased;
	}

	void Simulator::setBidirectionalBaseTime(const float t_time)
	{
		m_windBidirectionalBaseTime = t_time;
	}

	void Simulator::setBidirectionalR(const float t_R) {
		m_windBidirectionalR = t_R;
	}

	void Simulator::setWindAngle(const float t_windAngle)
	{
		m_firstWindAngle = t_windAngle;
		setWindDirection(t_windAngle);
	}

	void Simulator::setWindDirection(const float t_windAngle) {
		const float windAngle{ glm::radians(t_windAngle) };
		m_simulationParameters.windDirection = float2{ glm::cos(windAngle), glm::sin(windAngle) };
	}

	void Simulator::applyWindSpeed(const float t_windSpeed) {
		m_simulationParameters.windSpeed = t_windSpeed;
	}

	void Simulator::setWindSpeed(const float t_windSpeed)
	{
		m_windSpeed = t_windSpeed;
		applyWindSpeed(m_windSpeed);
	}

	void Simulator::setVenturiStrength(const float t_venturiStrength)
	{
		m_simulationParameters.venturiStrength = t_venturiStrength;
	}

	void Simulator::setWindWarpingMode(const WindWarpingMode t_windWarpingMode)
	{
		m_launchParameters.windWarpingMode = t_windWarpingMode;
	}

	void Simulator::setWindWarpingCount(const int t_windWarpingCount)
	{
		STHE_ASSERT(t_windWarpingCount >= 0, "Wind warping count must be greater than or equal to 0");
		STHE_ASSERT(t_windWarpingCount <= 4, "Wind warping count must be smaller than or equal to 4");

		m_launchParameters.windWarping.count = t_windWarpingCount;
	}

	void Simulator::setWindWarpingDivisor(const float t_windWarpingDivisor)
	{
		m_launchParameters.windWarping.i_divisor = 1.f / t_windWarpingDivisor;
	}

	void Simulator::setWindWarpingRadius(const int t_index, const float t_windWarpingRadius)
	{
		STHE_ASSERT(t_index >= 0, "Index must be greater than or equal to 0");
		STHE_ASSERT(t_index < 4, "Index must be smaller than 4");

		m_launchParameters.windWarping.radii[t_index] = t_windWarpingRadius;

		if (m_isAwake)
		{
			m_reinitializeWindWarping = true;
		}
	}

	void Simulator::setWindWarpingStrength(const int t_index, const float t_windWarpingStrength)
	{
		STHE_ASSERT(t_index >= 0, "Index must be greater than or equal to 0");
		STHE_ASSERT(t_index < 4, "Index must be smaller than 4");

		m_launchParameters.windWarping.strengths[t_index] = t_windWarpingStrength;
	}

	void Simulator::setWindWarpingGradientStrength(const int t_index, const float t_windWarpingGradientStrength)
	{
		STHE_ASSERT(t_index >= 0, "Index must be greater than or equal to 0");
		STHE_ASSERT(t_index < 4, "Index must be smaller than 4");

		m_launchParameters.windWarping.gradientStrengths[t_index] = t_windWarpingGradientStrength;
	}

	void Simulator::setWindShadowMode(const WindShadowMode t_windShadowMode)
	{
		m_launchParameters.windShadowMode = t_windShadowMode;
	}

	void Simulator::setWindShadowDistance(const float t_windShadowDistance)
	{
		m_simulationParameters.windShadowDistance = t_windShadowDistance;
	}

	void Simulator::setMinWindShadowAngle(const float t_minWindShadowAngle)
	{
		m_simulationParameters.minWindShadowAngle = glm::tan(glm::radians(t_minWindShadowAngle));
	}

	void Simulator::setMaxWindShadowAngle(const float t_maxWindShadowAngle)
	{
		m_simulationParameters.maxWindShadowAngle = glm::tan(glm::radians(t_maxWindShadowAngle));
	}

	void Simulator::setAbrasionStrength(const float t_abrasionStrength)
	{
		m_simulationParameters.abrasionStrength = t_abrasionStrength;
	}

	void Simulator::setSoilAbrasionStrength(const float t_abrasionStrength)
	{
		m_simulationParameters.soilAbrasionStrength = t_abrasionStrength;
	}

	void Simulator::setAbrasionThreshold(const float t_abrasionThreshold)
	{
		m_simulationParameters.abrasionThreshold = t_abrasionThreshold;
	}

	void Simulator::setSaltationMode(const SaltationMode t_saltationMode)
	{
		m_launchParameters.saltationMode = t_saltationMode;
	}

	void Simulator::setSaltationStrength(const float t_saltationStrength)
	{
		m_simulationParameters.saltationStrength = t_saltationStrength;
	}

	void Simulator::setReptationStrength(const float t_reptationStrength)
	{
		m_simulationParameters.reptationStrength = t_reptationStrength;
	}

	void Simulator::setReptationSmoothingStrength(const float t_reptationStrength)
	{
		m_simulationParameters.reptationSmoothingStrength = t_reptationStrength;
	}

	void Simulator::setReptationUseWindShadow(const float t_reptationUseWindShadow) {
		m_simulationParameters.reptationUseWindShadow = t_reptationUseWindShadow;
	}

	void Simulator::setBedrockAvalancheMode(const BedrockAvalancheMode t_bedrockAvalancheMode)
	{
		m_launchParameters.bedrockAvalancheMode = t_bedrockAvalancheMode;
	}

	void Simulator::setAvalancheIterations(const int t_avalancheIterations)
	{
		m_launchParameters.avalancheIterations = t_avalancheIterations;
	}

	void Simulator::setPressureProjectionIterations(int t_iters) {
		m_launchParameters.projection.jacobiIterations = t_iters;
	}

	void Simulator::setBedrockAvalancheIterations(const int t_bedrockAvalancheIterations)
	{
		m_launchParameters.bedrockAvalancheIterations = t_bedrockAvalancheIterations;
	}

	void Simulator::setSoilAvalancheIterations(const int t_soilAvalancheIterations)
	{
		m_launchParameters.soilAvalancheIterations = t_soilAvalancheIterations;
	}

	void Simulator::setAvalancheAngle(const float t_avalancheAngle)
	{
		m_simulationParameters.avalancheAngle = glm::tan(glm::radians(t_avalancheAngle));
	}

	void Simulator::setBedrockAngle(const float t_bedrockAngle)
	{
		m_simulationParameters.bedrockAngle = glm::tan(glm::radians(t_bedrockAngle));
	}

	void Simulator::setVegetationAngle(const float t_vegetationAngle)
	{
		m_simulationParameters.vegetationAngle = glm::tan(glm::radians(t_vegetationAngle));
	}

	void Simulator::setSoilAngle(const float t_soilAngle)
	{
		m_simulationParameters.soilAngle = glm::tan(glm::radians(t_soilAngle));
	}

	void Simulator::setVegetationSoilAngle(const float t_vegetationSoilAngle)
	{
		m_simulationParameters.vegetationSoilAngle = glm::tan(glm::radians(t_vegetationSoilAngle));
	}

	void Simulator::setWavePeriod(const float t_val) {
		m_simulationParameters.wavePeriod = t_val;
	}

	void Simulator::setWaveStrength(const float t_val) {
		m_simulationParameters.waveStrength = t_val;
	}

	void Simulator::setWaveDepthScale(const float t_val) {
		m_simulationParameters.waveDepthScale = t_val;
	}

	void Simulator::setSedimentCapacityConstant(const float t_val) {
		m_simulationParameters.sedimentCapacityConstant = t_val;
	}
	void Simulator::setSedimentDepositionConstant(const float t_val) {
		m_simulationParameters.sedimentDepositionConstant = t_val;
	}
	void Simulator::setSandDissolutionConstant(const float t_val) {
		m_simulationParameters.sandDissolutionConstant = t_val;
	}
	void Simulator::setSoilDissolutionConstant(const float t_val) {
		m_simulationParameters.soilDissolutionConstant = t_val;
	}
	void Simulator::setBedrockDissolutionConstant(const float t_val) {
		m_simulationParameters.bedrockDissolutionConstant = t_val;
	}

	void Simulator::setMoistureEvaporationScale(float v) {
		m_simulationParameters.moistureEvaporationScale = v;
	}
	void Simulator::setSandMoistureRate(float v) {
		m_simulationParameters.sandMoistureRate = v;
	}
	void Simulator::setSoilMoistureRate(float v) {
		m_simulationParameters.soilMoistureRate = v;
	}
	void Simulator::setTerrainThicknessMoistureThreshold(float v) {
		m_simulationParameters.iTerrainThicknessMoistureThreshold = v > 1e-6f ? 1.f / v : 0.f;
	}
	void Simulator::setMoistureCapacityConstant(float v) {
		m_simulationParameters.moistureCapacityConstant = v;
	}

	void Simulator::setWaterBorderLevel(float v) {
		m_simulationParameters.waterBorderLevel = v;
	}

	void Simulator::setWaterLevel(float v) {
		m_simulationParameters.waterLevel = v;
	}

	void Simulator::setEvaporationRate(float v) {
		m_simulationParameters.evaporationRate = v;
	}
	void Simulator::setRainStrength(float v) {
		m_simulationParameters.rainStrength = v;
	}
	void Simulator::setRainPeriod(float v) {
		m_simulationParameters.rainPeriod = v;
	}
	void Simulator::setRainScale(float v) {
		m_simulationParameters.rainScale = v;
	}
	void Simulator::setRainProbabilityMin(float v) {
		m_simulationParameters.rainProbabilityMin = v;
	}
	void Simulator::setRainProbabilityMax(float v) {
		m_simulationParameters.rainProbabilityMax = v;
	}
	void Simulator::setRainProbabilityHeightRange(float v) {
		m_simulationParameters.iRainProbabilityHeightRange = v > 1e-6f ? 1.f / v : 0.f;
	}

	void Simulator::setTimeMode(const TimeMode t_timeMode)
	{
		m_launchParameters.timeMode = t_timeMode;
	}

	void Simulator::setTimeScale(const float t_timeScale)
	{
		m_timeScale = t_timeScale;
	}

	void Simulator::setFixedDeltaTime(const float t_fixedDeltaTime)
	{
		m_fixedDeltaTime = t_fixedDeltaTime;
	}

	void Simulator::setInitializationParameters(const InitializationParameters& t_initializationParameters)
	{
		m_initializationParameters = t_initializationParameters;
	}

	void Simulator::setRenderParameters(const RenderParameters& t_renderParameters)
	{
		m_renderParameters = t_renderParameters;
		m_renderParameterBuffer->upload(reinterpret_cast<char*>(&m_renderParameters), sizeof(RenderParameters));
	}

	void Simulator::setCoverageThreshold(const float t_threshold)
	{
		m_coverageThreshold = t_threshold;
	}

	int Simulator::addVegType()
	{
		int i{ m_simulationParameters.vegTypeCount++ };

		m_vegPrefabs.files[i] = std::filesystem::path{};
		//m_vegTypes[i] = VegetationType{};
		
		std::fill(m_vegMatrix.begin() + i * c_maxVegTypeCount, m_vegMatrix.begin() + (i + 1) * c_maxVegTypeCount, 1.0f);

		if (m_isAwake)
		{
			m_vegMatrixBuffer.upload(m_vegMatrix);
		}

		return i;
	}

	void Simulator::setStopIterations(const int t_stopIterations) {
		m_stopIterations = t_stopIterations;
	}

	void Simulator::setVegetationType(const int t_index, const VegetationType& t_type)
	{
		while (t_index >= m_simulationParameters.vegTypeCount) {
			addVegType();
		}

		m_vegTypes.maxRadius[t_index] = t_type.maxRadius;
		m_vegTypes.growthRate[t_index] = t_type.growthRate;
		m_vegTypes.positionAdjustRate[t_index] = t_type.positionAdjustRate;
		m_vegTypes.damageRate[t_index] = t_type.damageRate;
		m_vegTypes.shrinkRate[t_index] = t_type.shrinkRate;
		m_vegTypes.maxMaturityTime[t_index] = t_type.maxMaturityTime;
		m_vegTypes.maturityPercentage[t_index] = t_type.maturityPercentage;
		m_vegTypes.height[t_index] = t_type.height;
		m_vegTypes.waterUsageRate[t_index] = t_type.waterUsageRate;
		m_vegTypes.waterStorageCapacity[t_index] = t_type.waterStorageCapacity;
		m_vegTypes.waterResistance[t_index] = t_type.waterResistance;
		m_vegTypes.minMoisture[t_index] = t_type.minMoisture;
		m_vegTypes.maxMoisture[t_index] = t_type.maxMoisture;
		m_vegTypes.soilCompatibility[t_index] = t_type.soilCompatibility;
		m_vegTypes.sandCompatibility[t_index] = t_type.sandCompatibility;
		m_vegTypes.terrainCoverageResistance[t_index] = t_type.terrainCoverageResistance;
		m_vegTypes.maxSlope[t_index] = t_type.maxSlope;
		m_vegTypes.baseSpawnRate[t_index] = t_type.baseSpawnRate;
		m_vegTypes.densitySpawnMultiplier[t_index] = t_type.densitySpawnMultiplier;
		m_vegTypes.windSpawnMultiplier[t_index] = t_type.windSpawnMultiplier;
		m_vegTypes.humusRate[t_index] = t_type.humusRate;
		m_vegTypes.lightConditions[t_index] = t_type.lightConditions;
		m_vegTypes.separation[t_index] = t_type.separation;

		m_uploadVegTypes = m_isAwake;
	}

	void Simulator::setVegetationTypeMesh(const int t_index, const std::filesystem::path& file)
	{
		if (m_vegPrefabs.gameObjects[t_index] != nullptr)
		{
			getScene().removeGameObject(*m_vegPrefabs.gameObjects[t_index]);
		}

		m_vegPrefabs.files[t_index] = file;

		if (m_isAwake)
		{
			std::cout << "Loading" << file.string() << std::endl;
			sthe::Importer importer{ file.string() };

			sthe::GameObject& gameObject{ importer.importModel(getScene(), m_vegPrefabs.program) };
			gameObject.getTransform().setParent(&getGameObject().getTransform(), false);

			m_vegPrefabs.gameObjects[t_index] = &gameObject;
			m_vegPrefabs.meshRenderers[t_index] = gameObject.getComponentsInChildren<sthe::MeshRenderer>();

			for (sthe::MeshRenderer* const meshRenderer : m_vegPrefabs.meshRenderers[t_index])
			{
				meshRenderer->setInstanceCount(m_launchParameters.vegCountsPerType[t_index]);
			}

			for (auto& material : importer.getMaterials())
			{
				material->setBuffer(GL_SHADER_STORAGE_BUFFER, STHE_STORAGE_BUFFER_CUSTOM0, m_vegPrefabs.buffer);
				material->setBuffer(GL_SHADER_STORAGE_BUFFER, STHE_STORAGE_BUFFER_CUSTOM0 + 1, m_vegPrefabs.mapBuffer);
			}
		}
	}

	std::vector<int> Simulator::getVegCount() const {
		std::vector<int> vegCount(1 + c_maxVegTypeCount);
		vegCount[0] = m_launchParameters.vegCount;
		for (int i = 0; i < c_maxVegTypeCount; ++i) {
			vegCount[i + 1] = m_launchParameters.vegCountsPerType[i];
		}
		return vegCount;
	}

	void Simulator::setVegMatrix(const std::vector<float>& vegMatrix)
	{
		m_vegMatrix = vegMatrix;

		if (m_isAwake)
		{
			m_vegMatrixBuffer.upload(vegMatrix);
		}
	}

	// Getters
	int Simulator::getTimeStep() const {
		return m_timeStep;
	}

	int Simulator::getPerfStep() const {
		return m_perfStep;
	}

	int Simulator::getVegTypeCount() const
	{
		return m_simulationParameters.vegTypeCount;
	}

	bool Simulator::isPaused() const
	{
		return m_isPaused;
	}

	std::shared_ptr<sthe::gl::Texture2D> Simulator::getTerrainMap() const {
		return m_terrainMap;
	}
	std::shared_ptr<sthe::gl::Texture2D> Simulator::getResistanceMap() const {
		return m_resistanceMap;
	}

}

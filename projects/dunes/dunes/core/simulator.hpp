#pragma once

#include <dunes/core/water.hpp>
#include <dunes/components/water_renderer.hpp>
#include "simulation_parameters.hpp"
#include "launch_parameters.hpp"
#include "render_parameters.hpp"
#include <cufft.h>
#include <sthe/sthe.hpp>
#include <sthe/gl/buffer.hpp>
#include <vector>

namespace dunes
{

	class Simulator : public sthe::Component
	{
	public:
		// Constructors
		Simulator();
		Simulator(const Simulator& t_simulator) = delete;
		Simulator(Simulator&& t_simulator) = default;

		// Destructor
		~Simulator();

		// Operators
		Simulator& operator=(const Simulator& t_simulator) = delete;
		Simulator& operator=(Simulator&& t_simulator) = default;

		// Functionality
		void reinitialize(const glm::ivec2& t_gridSize, const float t_gridScale);
		void awake();
		void update();
		void resume();
		void pause();

		void setupCoverageCalculation();
		void calculateCoverage();
		void cleanupCoverageCalculation();

		// Setters
		void setUseBilinear(const bool t_useBilinear);
		void setWindAngle(const float t_windAngle);
		void setWindSpeed(const float t_windSpeed);
		void setVenturiStrength(const float t_venturiStrength);
		void setWindWarpingMode(const WindWarpingMode t_windWarpingMode);
		void setWindWarpingCount(const int t_windWarpingCount);
		void setWindWarpingDivisor(const float t_windWarpingDivisor);
		void setWindWarpingRadius(const int t_index, const float t_windWarpingRadius);
		void setWindWarpingStrength(const int t_index, const float t_windWarpingStrength);
		void setWindWarpingGradientStrength(const int t_index, const float t_windWarpingGradientStrength);
		void setWindShadowMode(const WindShadowMode t_windShadowMode);
		void setWindShadowDistance(const float t_windShadowDistance);
		void setMinWindShadowAngle(const float t_minWindShadowAngle);
		void setMaxWindShadowAngle(const float t_maxWindShadowAngle);
		void setAbrasionStrength(const float t_abrasionStrength);
		void setSoilAbrasionStrength(const float t_abrasionStrength);
		void setAbrasionThreshold(const float t_abrasionThreshold);
		void setSaltationMode(const SaltationMode t_saltationMode);
		void setSaltationStrength(const float t_saltationStrength);
		void setReptationStrength(const float t_reptationStrength);
		void setReptationSmoothingStrength(const float t_reptationStrength);
		void setReptationUseWindShadow(const float t_reptationUseWindShadow);

		void setBedrockAvalancheMode(const BedrockAvalancheMode t_bedrockAvalancheMode);
		void setAvalancheIterations(const int t_avalancheIterations);
		void setPressureProjectionIterations(int t_iters);

		void setBedrockAvalancheIterations(const int t_bedrockAvalancheIterations);
		void setSoilAvalancheIterations(const int t_soilAvalancheIterations);

		void setAvalancheAngle(const float t_avalancheAngle);
		void setBedrockAngle(const float t_bedrockAngle);
		void setVegetationAngle(const float t_vegetationAngle);
		void setSoilAngle(const float t_Angle);
		void setVegetationSoilAngle(const float t_Angle);

		void setWavePeriod(const float t_val);
		void setWaveStrength(const float t_val);
		void setWaveDepthScale(const float t_val);

		void setSedimentCapacityConstant( const float t_val );
		void setSedimentDepositionConstant( const float t_val );
		void setSandDissolutionConstant( const float t_val );
		void setSoilDissolutionConstant( const float t_val );
		void setBedrockDissolutionConstant( const float t_val );

		void setWaterBorderLevel(float v);
		void setWaterLevel(float v);

		void resetPerformanceAverages();

		void setMoistureEvaporationScale(float v);
		void setSandMoistureRate(float v);
		void setSoilMoistureRate(float v);
		void setTerrainThicknessMoistureThreshold(float v);
		void setMoistureCapacityConstant(float v);

		void setEvaporationRate(float v);
		void setRainStrength(float v);
		void setRainPeriod(float v);
		void setRainScale(float v);
		void setRainProbabilityMin(float v);
		void setRainProbabilityMax(float v);
		void setRainProbabilityHeightRange(float v);

		void setTimeMode(const TimeMode t_timeMode);
		void setTimeScale(const float t_timeScale);
		void setFixedDeltaTime(const float t_fixedDeltaTime);
		void setInitializationParameters(const InitializationParameters& t_initializationParameters);
		void setRenderParameters(const RenderParameters& t_renderParameters);
		void setCoverageThreshold(const float t_threshold);
		float getCoverage();
		void setTargetCoverage(const float t_targetCoverage);
		void setCoverageSpawnAmount(const float t_amount);
		void setCoverageSubtractAmount(const float t_amount);
		void setCoverageSpawnUniform(const bool t_uniform);
		void setCoverageRadius(const int t_radius);
		void setSpawnSteps(const int t_steps);
		void setConstantCoverage(const bool t_constantCoverage);
		void setConstantCoverageAllowRemove(const bool t_constantCoverageAllowRemove);
		void setProjectionMode(const ProjectionMode t_mode);

		const std::vector<float>& getWatchTimings();
		const std::vector<float>& getMeanWatchTimings();

		void setSecondWindAngle(const float t_windAngle);
		void enableBidirectional(const bool t_enable);
		void setBidirectionalStrengthBased(const bool t_sBased);
		void setBidirectionalBaseTime(const float t_time);
		void setBidirectionalR(const float t_R);
		void setMaxVegCount(float c);

		int addVegType();
		void setStopIterations(const int t_stopIterations);
		void setVegetationType(const int t_index, const VegetationType& t_type);
		void setVegetationTypeMesh(const int t_index, const std::filesystem::path& file);
		void setVegMatrix(const std::vector<float>& vegMatrix);

		void updateWindShadow();

		bool queryTimeStepHappened();

		// Getters
		std::vector<int> getVegCount() const;
		int getTimeStep() const;
		int getPerfStep() const;
		int getVegTypeCount() const;
		bool isPaused() const;
		std::shared_ptr<sthe::gl::Texture2D> getTerrainMap() const;
		std::shared_ptr<sthe::gl::Texture2D> getResistanceMap() const;
	private:
		// Functionality
		void setupLaunchParameters();
		void setupTerrain();
		void setupArrays();
		void setupBuffers();
		void setupWindWarping();
		void setupProjection();
		void setupVegPrefabs();
		void setupAdaptiveGrid();
		void map();
		void unmap();

		void setWindDirection(const float t_windAngle);
		void applyWindSpeed(const float t_windSpeed);

		bool m_timestepHappened{ false };

		// Attributes
		SimulationParameters m_simulationParameters;
		LaunchParameters m_launchParameters;
		InitializationParameters m_initializationParameters;
		RenderParameters m_renderParameters;
		float m_timeScale;
		float m_fixedDeltaTime;
		float m_coverage;
		float m_coverageThreshold;
		bool m_constantCoverage{ false };
		bool m_constantCoverageAllowRemove{ false };
		float m_targetCoverage{ 0.5f };
		float m_coverageSpawnAmount{ 0.01f };
		float m_coverageSubtractAmount{ 0.01f };
		int m_coverageRadius{ 100 };
		bool m_coverageSpawnUniform{ false };
		int m_spawnSteps{ 10 };
		int m_timeStep = 0;
		int m_perfStep = 0;
		int m_stopIterations = 0;

		float m_time{ 0.f };
		float m_firstWindAngle{ 0.0f };
		float m_windSpeed{ 10.f };
		float m_secondWindAngle{ 0.0f };
		float m_windBidirectionalR{ 1.f };
		float m_windBidirectionalBaseTime{ 1.f };
		bool m_enableBidirectional{ false };
		bool m_bidirectionalStrengthBased{ true };

		sthe::TerrainRenderer* m_terrainRenderer;
		std::shared_ptr<sthe::Terrain> m_terrain;
		std::shared_ptr<sthe::CustomMaterial> m_material;
		std::shared_ptr<sthe::gl::Program> m_program;
		std::shared_ptr<sthe::CustomMaterial> m_rimMaterial;
		std::shared_ptr<sthe::gl::Program> m_rimProgram;

		WaterRenderer* m_waterRenderer;
		std::shared_ptr<Water> m_water;
		std::shared_ptr<sthe::CustomMaterial> m_waterMaterial;
		std::shared_ptr<sthe::gl::Program> m_waterProgram;
		std::shared_ptr<sthe::CustomMaterial> m_waterRimMaterial;
		std::shared_ptr<sthe::gl::Program> m_waterRimProgram;

		std::shared_ptr<sthe::gl::Texture2D> m_terrainMap;
		std::shared_ptr<sthe::gl::Texture2D> m_windMap;
		std::shared_ptr<sthe::gl::Texture2D> m_resistanceMap;
		std::shared_ptr<sthe::gl::Texture2D> m_fluxMap; // TODO: Needed as texture?
		std::shared_ptr<sthe::gl::Texture2D> m_sedimentMap;
		std::shared_ptr<sthe::gl::Texture2D> m_terrainMoistureMap;
		std::shared_ptr<sthe::gl::Texture2D> m_vegetationHeightMap;
		std::shared_ptr<sthe::gl::Texture2D> m_shadowMap;
		std::shared_ptr<sthe::gl::Buffer> m_renderParameterBuffer;

		sthe::cu::Array2D m_terrainArray;
		sthe::cu::Array2D m_windArray;
		sthe::cu::Array2D m_resistanceArray;
		sthe::cu::Array2D m_fluxArray;
		sthe::cu::Array2D m_terrainMoistureArray;
		sthe::cu::Array2D m_sedimentArray;
		sthe::cu::Array2D m_vegetationHeightArray;
		sthe::cu::Array2D m_shadowArray;
		sthe::cu::Buffer m_slabBuffer;
		sthe::cu::Buffer m_tmpBuffer;
		sthe::cu::Buffer m_vegBuffer;
		sthe::cu::Buffer m_vegMapBuffer;
		sthe::cu::Buffer m_vegCountBuffer;
		sthe::cu::Buffer m_vegTypeBuffer;
		sthe::cu::Buffer m_vegMatrixBuffer;
		sthe::cu::Buffer m_seedBuffer;
		std::array<sthe::cu::Buffer, 2> m_windWarpingBuffers;
		std::unique_ptr<sthe::cu::Buffer> m_coverageMap;
		sthe::cu::Buffer m_velocityBuffer;
		cudaTextureDesc m_textureDescriptor;

		bool m_isAwake;
		bool m_isPaused;
		bool m_reinitializeWindWarping;
		bool m_uploadVegTypes;

		std::vector<float> m_watchTimings;
		std::vector<float> m_meanWatchTimings;
		std::vector<sthe::cu::Stopwatch> m_watches;

		struct VegetationPrefabs 
		{
			std::shared_ptr<sthe::gl::Program> program;
			std::shared_ptr<sthe::gl::Buffer> buffer;
			std::shared_ptr<sthe::gl::Buffer> mapBuffer;
			std::array<sthe::GameObject*, c_maxVegTypeCount> gameObjects{};
			std::array<std::vector<sthe::MeshRenderer*>, c_maxVegTypeCount> meshRenderers{};
			std::array<std::filesystem::path, c_maxVegTypeCount> files{};
		} m_vegPrefabs;

		VegetationTypeSoA m_vegTypes{};
		std::vector<float> m_vegMatrix;

		struct AdaptiveGrid
		{
			sthe::cu::Buffer gridBuffer;
			sthe::cu::Buffer keyBuffer;
			sthe::cu::Buffer indexBuffer;
			sthe::cu::Buffer vegBuffer;
		} m_adaptiveGrid;
	};
}

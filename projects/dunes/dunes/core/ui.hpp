#pragma once

#include "simulator.hpp"
#include "../util/io.hpp"
#include <sthe/sthe.hpp>

namespace dunes
{

	class UI : public sthe::Component
	{
	public:
		// Constructors
		UI() = default;
		UI(const UI& t_ui) = default;
		UI(UI&& t_ui) = default;

		// Destructor
		~UI() = default;

		// Operators
		UI& operator=(const UI& t_ui) = default;
		UI& operator=(UI&& t_ui) = default;

		// Functionality
		void awake();
		void onGUI();

		bool render_vegetation{ true };
	private:
		// Static
		static inline const char* saltationModes[2]{ "Backward", "Forward" };
		static inline const char* windWarpingModes[2]{ "None", "Standard" };
		static inline const char* windShadowModes[2]{ "Linear", "Curved" };
		static inline const char* bedrockAvalancheModes[2]{ "To Sand", "To Bedrock" };
		static inline const char* timeModes[2]{ "Delta Time", "Fixed Delta Time" };
		static inline const char* initializationTargets[NumNoiseGenerationTargets]{ "Bedrock", "Sand", "Vegetation", "Abrasion Resistance", "Soil", "Water", "Moisture"};
		static inline const char* watchTimingNames[20]{ 
			"All CUDA", 
			"All Vegetation",
			"Veg. Datastructure",
			"Veg. Growth",
			"Veg. Raster",
			"Veg. ShadowMap",
			"All Terrain & Wind",
			"Venturi", 
			"Wind Warping", 
			"Pressure Projection",
			"Wind Shadow", 
			"Saltation", 
			"Reptation", 
			"Sand Avalanching", 
			"Bedrock & Soil Avalanching", 
			"All Water & Sediment",
			"Rain",
			"Water Transport",
			"Moisture",
			"Sediment"
		};
		static inline const char* projectionModes[3]{ "None", "Jacobi", "FFT" };
		std::array<std::string, c_maxVegTypeCount> vegetationNames{ "Type 0", "Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6", "Type 7" };

		void initializeAll();

		// Functionality
		void createPerformanceNode();
		void createApplicationNode();
		void createRenderingNode();
		void createSceneNode();
		void createSimulationNode();

		// Attributes
		Simulator* m_simulator{ nullptr };

		float m_mean_frametime{ 0.f };
		float m_frametime{ 0.f };
		bool m_recordNextFrametime{ false };

		// Files
		bool toJson(const std::string& path);
		bool fromJson(const std::string& path);
		bool m_exportMaps = false;
		bool loadEXR(std::shared_ptr<sthe::gl::Texture2D> map, const std::string& input);
		std::string m_heightMapPath{};
		std::string m_resistanceMapPath{};

		// Application
		bool m_takeScreenshot{ false };
		std::string m_screenShotPath{};
		bool m_vSync{ false };
		bool m_calcCoverage{ false };
		float m_coverageThreshold{ 0.1f };
		int m_targetFrameRate{ 0 };
		bool m_constantCoverage{ false };
		bool m_constantCoverageAllowRemove{ false };
		float m_targetCoverage{ 1.0f };
		float m_coverageSpawnAmount{ 1.f };
		float m_coverageSubtractAmount{ 1.f };
		int m_coverageRadius{ 100 };
		bool m_coverageSpawnUniform{ false };
		int m_spawnSteps{ 10 };
		int m_stopIterations{ 0 };
		int maxVegCount{ 1000000 };

		// Simulation
		bool m_useBilinear{ true };
		glm::ivec2 m_gridSize{ 2048, 2048 };
		float m_gridScale{ 1.0f };

		float m_windAngle{ 0.0f };
		float m_secondWindAngle{ 45.0f };
		float m_windBidirectionalR{ 2.f };
		float m_windBidirectionalBaseTime{ 15.f };
		bool m_enableBidirectional{ false };
		bool m_bidirectionalStrengthBased{ true };
		float m_windSpeed{ 10.0f };

		float m_venturiStrength{ 0.005f };

		int m_windWarpingMode{ static_cast<int>(WindWarpingMode::None) };
		int m_windWarpingCount{ 2 };
		float m_windWarpingDivisor{ 1.0f };
		std::array<float, 4> m_windWarpingRadii{ 200.0f, 50.0f, 0.0f, 0.0f };
		std::array<float, 4> m_windWarpingStrengths{ 0.8f, 0.2f, 0.0f, 0.0f };
		std::array<float, 4> m_windWarpingGradientStrengths{ 30.f, 5.f, 0.0f, 0.0f };

		int m_windShadowMode{ static_cast<int>(WindShadowMode::Linear) };
		float m_windShadowDistance{ 10.0f };
		float m_minWindShadowAngle{ 10.0f };
		float m_maxWindShadowAngle{ 15.0f };

		float m_abrasionStrength{ 0.0f };
		float m_soilAbrasionStrength{ 0.0f };
		float m_abrasionThreshold{ 0.025f };
		int m_saltationMode{ static_cast<int>(SaltationMode::Forward) };
		float m_saltationStrength{ 1.f };
		float m_reptationStrength{ 0.0f };
		float m_reptationSmoothingStrength{ 0.0f };
		bool m_reptationUseWindShadow{ false };

		int m_bedrockAvalancheMode{ static_cast<int>(BedrockAvalancheMode::ToSand) };
		int m_avalancheIterations{ 50 };
		int m_pressureProjectionIterations{ 50 };
		int m_projectionMode{ static_cast<int>(ProjectionMode::FFT) };
		int m_bedrockAvalancheIterations{ 1 };
		int m_soilAvalancheIterations{ 1 };
		float m_avalancheAngle{ 33.0f };
		float m_bedrockAngle{ 68.0f };
		float m_vegetationAngle{ 45.0f };
		float m_soilAngle{ 45.0f };
		float m_vegetationSoilAngle{ 68.0f };

		float m_wavePeriod{ 0.02f };
		float m_waveStrength{ 0.005f };
		float m_waveDepthScale{ 0.1f };

		float m_sedimentCapacityConstant{ 0.1f };
		float m_sedimentDepositionConstant{ 0.1f };
		float m_sandDissolutionConstant{ 0.1f };
		float m_soilDissolutionConstant{ 0.05f };
		float m_bedrockDissolutionConstant{ 0.01f };

		float m_waterBorderLevel{ 20.f };
		float m_waterLevel{ 0.f };

		float m_moistureEvaporationScale{ 0.05f };
		float m_sandMoistureRate{ 0.1f };
		float m_soilMoistureRate{ 0.02f };
		float m_terrainThicknessMoistureThreshold{ 1.f };
		float m_moistureCapacityConstant{ 1.f };

		float m_evaporationRate{ 0.01f };
		float m_rainStrength{0.001f};
		float m_rainPeriod{0.2f};
		float m_rainScale{30.f};
		float m_rainProbabilityMin{0.5f};
		float m_rainProbabilityMax{1.f};
		float m_rainProbabilityHeightRange{1000.f};

		int m_timeMode{ static_cast<int>(TimeMode::FixedDeltaTime) };
		float m_timeScale{ 1.0f };
		float m_fixedDeltaTime{ 0.5f };

		InitializationParameters m_initializationParameters{};
		RenderParameters m_renderParameters{};

		std::array<VegetationType, c_maxVegTypeCount> m_vegTypes
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

		int m_vegTypeCount{ 3 };
		std::array<std::string, c_maxVegTypeCount> m_vegMeshes{ dunes::getResourcePath() + "models\\MapleFall.obj",
			                                                    dunes::getResourcePath() + "models\\BushFlowerSmall.obj",
			                                                    dunes::getResourcePath() + "models\\seaweed.obj" };

		std::vector<float> m_vegMatrix;
	};

}

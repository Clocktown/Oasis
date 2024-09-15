#include "ui.hpp"
#include "simulator.hpp"
#include <sthe/sthe.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <tinyfiledialogs/tinyfiledialogs.h>
#include <tinyexr.h>

#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace dunes
{

	void UI::initializeAll() {
		sthe::getApplication().setVSyncCount(m_vSync);
		sthe::getApplication().setTargetFrameRate(m_targetFrameRate);

		m_simulator->pause();
		m_simulator->setUseBilinear(m_useBilinear);
		m_simulator->setInitializationParameters(m_initializationParameters);
		m_simulator->setRenderParameters(m_renderParameters);

		m_simulator->setStopIterations(m_stopIterations);
		m_simulator->setConstantCoverage(m_constantCoverage);
		m_simulator->setConstantCoverageAllowRemove(m_constantCoverageAllowRemove);
		m_simulator->setTargetCoverage(m_targetCoverage);
		m_simulator->setCoverageSpawnAmount(m_coverageSpawnAmount);
		m_simulator->setCoverageSubtractAmount(m_coverageSubtractAmount);
		m_simulator->setCoverageRadius(m_coverageRadius);
		m_simulator->setCoverageSpawnUniform(m_coverageSpawnUniform);
		m_simulator->setSpawnSteps(m_spawnSteps);
		m_simulator->setCoverageThreshold(m_coverageThreshold);

		m_simulator->setWindSpeed(m_windSpeed);
		m_simulator->setWindAngle(m_windAngle);
		m_simulator->setVenturiStrength(m_venturiStrength);
		m_simulator->enableBidirectional(m_enableBidirectional);
		m_simulator->setBidirectionalStrengthBased(m_bidirectionalStrengthBased);
		m_simulator->setSecondWindAngle(m_secondWindAngle);
		m_simulator->setBidirectionalR(m_windBidirectionalR);
		m_simulator->setBidirectionalBaseTime(m_windBidirectionalBaseTime);

		m_simulator->setWindWarpingMode(static_cast<WindWarpingMode>(m_windWarpingMode));
		m_simulator->setWindWarpingCount(m_windWarpingCount);
		m_simulator->setWindWarpingDivisor(m_windWarpingDivisor);
		for (int i{ 0 }; i < 4; ++i)
		{
			m_simulator->setWindWarpingStrength(i, m_windWarpingStrengths[i]);
		}
		for (int i{ 0 }; i < 4; ++i)
		{
			m_simulator->setWindWarpingGradientStrength(i, m_windWarpingGradientStrengths[i]);
		}
		for (int i{ 0 }; i < 4; ++i)
		{
			m_simulator->setWindWarpingRadius(i, m_windWarpingRadii[i]);
		}

		m_simulator->setWindShadowMode(static_cast<WindShadowMode>(m_windShadowMode));
		m_simulator->setWindShadowDistance(m_windShadowDistance);
		m_simulator->setMinWindShadowAngle(m_minWindShadowAngle);
		m_simulator->setMaxWindShadowAngle(m_maxWindShadowAngle);

		m_simulator->setSaltationMode(static_cast<SaltationMode>(m_saltationMode));
		m_simulator->setSaltationStrength(m_saltationStrength);
		m_simulator->setAbrasionStrength(m_abrasionStrength);
		m_simulator->setAbrasionThreshold(m_abrasionThreshold);
		m_simulator->setReptationStrength(m_reptationStrength);
		m_simulator->setReptationSmoothingStrength(m_reptationSmoothingStrength);
		m_simulator->setReptationUseWindShadow(float(m_reptationUseWindShadow));

		m_simulator->setAvalancheIterations(m_avalancheIterations);
		m_simulator->setPressureProjectionIterations(m_pressureProjectionIterations);
		m_simulator->setProjectionMode(static_cast<ProjectionMode>(m_projectionMode));
		m_simulator->setAvalancheAngle(m_avalancheAngle);
		m_simulator->setVegetationAngle(m_vegetationAngle);
		m_simulator->setBedrockAvalancheMode(static_cast<BedrockAvalancheMode>(m_bedrockAvalancheMode));
		m_simulator->setBedrockAvalancheIterations(m_bedrockAvalancheIterations);
		m_simulator->setBedrockAngle(m_bedrockAngle);

		m_simulator->setSoilAvalancheIterations(m_soilAvalancheIterations);
		m_simulator->setSoilAngle(m_soilAngle);
		m_simulator->setVegetationSoilAngle(m_vegetationSoilAngle);

		m_simulator->setWavePeriod(m_wavePeriod);
		m_simulator->setWaveStrength(m_waveStrength);
		m_simulator->setWaveDepthScale(m_waveDepthScale);

		m_simulator->setSedimentCapacityConstant(m_sedimentCapacityConstant);
		m_simulator->setSedimentDepositionConstant(m_sedimentDepositionConstant);
		m_simulator->setSandDissolutionConstant(m_sandDissolutionConstant);
		m_simulator->setSoilDissolutionConstant(m_soilDissolutionConstant);
		m_simulator->setBedrockDissolutionConstant(m_bedrockDissolutionConstant);

		m_simulator->setMoistureEvaporationScale(m_moistureEvaporationScale);
		m_simulator->setSandMoistureRate(m_sandMoistureRate);
		m_simulator->setSoilMoistureRate(m_soilMoistureRate);
		m_simulator->setTerrainThicknessMoistureThreshold(m_terrainThicknessMoistureThreshold);
		m_simulator->setMoistureCapacityConstant(m_moistureCapacityConstant);
		m_simulator->setWaterBorderLevel(m_waterBorderLevel);
		m_simulator->setWaterLevel(m_waterLevel);

		m_simulator->setEvaporationRate(m_evaporationRate);
		m_simulator->setRainStrength(m_rainStrength);
		m_simulator->setRainPeriod(m_rainPeriod);
		m_simulator->setRainScale(m_rainScale);
		m_simulator->setRainProbabilityMin(m_rainProbabilityMin);
		m_simulator->setRainProbabilityMax(m_rainProbabilityMax);
		m_simulator->setRainProbabilityHeightRange(m_rainProbabilityHeightRange);

		m_simulator->setTimeMode(static_cast<TimeMode>(m_timeMode));
		m_simulator->setTimeScale(m_timeScale);
		m_simulator->setFixedDeltaTime(m_fixedDeltaTime);

		m_vegMatrix.resize(c_maxVegTypeCount * c_maxVegTypeCount, 1.0f);
		m_vegMatrix[1 + 0 * c_maxVegTypeCount] = 0.5f;
		m_vegMatrix[0 + 1 * c_maxVegTypeCount] = 2.0f;

		for (int i = 0; i < c_maxVegTypeCount; ++i)
		{
			m_simulator->setVegetationType(i, m_vegTypes[i]);
			m_simulator->setVegetationTypeMesh(i, m_vegMeshes[i]);
		}

		m_simulator->reinitialize(m_gridSize, m_gridScale);

		if (m_calcCoverage) {
			m_simulator->setupCoverageCalculation();
		}
		m_simulator->setVegMatrix(m_vegMatrix);
	}

	// Functionality
	void UI::awake()
	{
		m_simulator = getGameObject().getComponent<Simulator>();

		STHE_ASSERT(m_simulator != nullptr, "Simulator cannot be nullptr");

		initializeAll();
	}

	void UI::onGUI()
	{
		m_frametime = sthe::getApplication().getUnscaledDeltaTime();
		const int N = m_simulator->getTimeStep();
		if (m_recordNextFrametime) {
			m_mean_frametime = ((m_mean_frametime * (N - 1)) + m_frametime) / N;
			m_recordNextFrametime = false;
		}
		if (m_simulator->queryTimeStepHappened()) {
			m_recordNextFrametime = true;
		}
		if (m_takeScreenshot) {
			m_takeScreenshot = false;

			glm::ivec2 res = sthe::getWindow().getResolution();

			std::vector<uint8_t> screen_pixels(3 * res.x * res.y);

			// Generates pixel path performance warning, but this is fine in our scenario
			glReadPixels(0,
				0,
				res.x,
				res.y,
				GL_RGB,
				GL_UNSIGNED_BYTE,
				screen_pixels.data());

			stbi_write_png_compression_level = 9;
			stbi_flip_vertically_on_write(true);

			stbi_write_png(m_screenShotPath.c_str(),
				res.x,
				res.y,
				3,
				screen_pixels.data(),
				3 * res.x);
		}
		ImGui::Begin("Settings");

		if (ImGui::Button("Screenshot")) {
			char const* filterPatterns[1] = { "*.png" };
			auto output = tinyfd_saveFileDialog("Save Screenshot", "./screenshot.png", 1, filterPatterns, "Portable Network Graphics (.png)");
			if (output != nullptr) {
				m_takeScreenshot = true;
				m_screenShotPath = output;
			}
		}

		createPerformanceNode();
		createApplicationNode();
		createRenderingNode();
		createSceneNode();
		createSimulationNode();

		ImGui::End();
	}

	void UI::createPerformanceNode() {
		if (ImGui::TreeNode("Performance")) {
			ImGui::LabelText("Frametime [ms]", "%f", 1000.f * sthe::getApplication().getUnscaledDeltaTime());
			const auto& times = m_simulator->getWatchTimings();
			const auto& meanTimes = m_simulator->getMeanWatchTimings();
			for (int i = 0; i < times.size(); ++i) {
				std::string ltext = watchTimingNames[i];
				ltext += " [ms]";
				ImGui::LabelText(ltext.c_str(), "%f", times[i]);
			}
			ImGui::Separator();
			for (int i = 0; i < meanTimes.size(); ++i) {
				std::string ltext = "Avg. ";
				ltext += watchTimingNames[i];
				ltext += " [ms]";
				ImGui::LabelText(ltext.c_str(), "%f", meanTimes[i]);
			}
			ImGui::LabelText("Avg. Frametime [ms]", "%f", 1000.f * m_mean_frametime);
			ImGui::TreePop();
		}
	}

	void UI::createApplicationNode()
	{
		sthe::Application& application{ sthe::getApplication() };
		if (ImGui::TreeNode("Application"))
		{
			if (ImGui::Checkbox("VSync", &m_vSync))
			{
				application.setVSyncCount(m_vSync);
			}

			if (ImGui::DragInt("Target Frame Rate", &m_targetFrameRate))
			{
				application.setTargetFrameRate(m_targetFrameRate);
			}

			ImGui::TreePop();
		}
	}

	void UI::createRenderingNode()
	{
		if (ImGui::TreeNode("Rendering"))
		{
			bool dirty = false;
			ImGui::Checkbox("Render Vegetation", &render_vegetation);
			dirty |= ImGui::ColorEdit4("Sand Color", glm::value_ptr(m_renderParameters.sandColor));
			dirty |= ImGui::ColorEdit4("Soil Color", glm::value_ptr(m_renderParameters.soilColor));
			dirty |= ImGui::ColorEdit4("Humus Color", glm::value_ptr(m_renderParameters.humusColor));
			dirty |= ImGui::ColorEdit4("Vegetation Color", glm::value_ptr(m_renderParameters.vegetationColor));
			dirty |= ImGui::ColorEdit4("Bedrock Color", glm::value_ptr(m_renderParameters.bedrockColor));
			dirty |= ImGui::ColorEdit4("Water Color", glm::value_ptr(m_renderParameters.waterColor));
			dirty |= ImGui::ColorEdit4("Wind Shadow Color", glm::value_ptr(m_renderParameters.windShadowColor));
			dirty |= ImGui::ColorEdit4("Wet Color", glm::value_ptr(m_renderParameters.wetColor));
			dirty |= ImGui::DragFloat("Shadow/AO Strength", &m_renderParameters.shadowStrength, 0.01f, 0.f, 1.f);

			if (dirty) {
				m_simulator->setRenderParameters(m_renderParameters);
			}
			ImGui::TreePop();
		}
	}

	bool SaveEXR(const float* rgba, int width, int height, const char* outfilename, int tinyexr_pixeltype) {

		EXRHeader header;
		InitEXRHeader(&header);

		EXRImage image;
		InitEXRImage(&image);

		image.num_channels = 4;

		std::vector<float> images[4];
		images[0].resize(width * height);
		images[1].resize(width * height);
		images[2].resize(width * height);
		images[3].resize(width * height);

		// Split RGBRGBRGB... into R, G, B and A layer
		for (int i = 0; i < width * height; i++) {
			images[0][i] = rgba[4 * i + 0];
			images[1][i] = rgba[4 * i + 1];
			images[2][i] = rgba[4 * i + 2];
			images[3][i] = rgba[4 * i + 3];
		}

		float* image_ptr[4];
		image_ptr[0] = &(images[3].at(0)); // A
		image_ptr[1] = &(images[2].at(0)); // B
		image_ptr[2] = &(images[1].at(0)); // G
		image_ptr[3] = &(images[0].at(0)); // R

		image.images = (unsigned char**)image_ptr;
		image.width = width;
		image.height = height;

		header.num_channels = 4;
		header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
		// Must be (A)BGR order, since most of EXR viewers expect this channel order.
		strncpy(header.channels[0].name, "A", 255); header.channels[0].name[strlen("A")] = '\0';
		strncpy(header.channels[1].name, "B", 255); header.channels[1].name[strlen("B")] = '\0';
		strncpy(header.channels[2].name, "G", 255); header.channels[2].name[strlen("G")] = '\0';
		strncpy(header.channels[3].name, "R", 255); header.channels[3].name[strlen("R")] = '\0';

		header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
		header.requested_pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
		for (int i = 0; i < header.num_channels; i++) {
			header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
			header.requested_pixel_types[i] = tinyexr_pixeltype; // pixel type of output image to be stored in .EXR
		}

		const char* err = nullptr; // or nullptr in C++11 or later.
		int ret = SaveEXRImageToFile(&image, &header, outfilename, &err);
		free(header.channels);
		free(header.pixel_types);
		free(header.requested_pixel_types);
		if (ret != TINYEXR_SUCCESS) {
			std::string errorMsg = std::string("Could not save file:\n") + outfilename + "\nReason: " + err;
			tinyfd_messageBox("Error", errorMsg.c_str(), "ok", "error", 1);
			FreeEXRErrorMessage(err); // free's buffer for an error message
			return ret;
		}
	}

	bool UI::loadEXR(std::shared_ptr<sthe::gl::Texture2D> map, const std::string& input) {
		float* out; // width * height * RGBA
		int width;
		int height;
		const char* err = nullptr; // or nullptr in C++11

		int ret = LoadEXR(&out, &width, &height, input.c_str(), &err);
		
		if (ret != TINYEXR_SUCCESS) {
			if (err) {
				std::string errorMsg = std::string("Could not load file:\n") + input + "\nReason: " + err;
				tinyfd_messageBox("Error", errorMsg.c_str(), "ok", "error", 1);
				FreeEXRErrorMessage(err); // release memory of error message.
			}
		}
		else {
			if (width != m_gridSize.x || height != m_gridSize.y) {
				m_gridSize = { width, height };
				m_simulator->setInitializationParameters(m_initializationParameters);
				m_simulator->reinitialize(m_gridSize, m_gridScale);
			}
			map->upload(out, width, height, GL_RGBA, GL_FLOAT);
			free(out); // release memory of image data
			return true;
		}

		
		return false;
	}

	int getIndexFromNamedArray(const char** arr, int length, const std::string& val, int default_index) {
		int idx = std::find(&arr[0], &arr[length], val) - &arr[0];
		return idx == length ? default_index : idx;
	}

	bool UI::fromJson(const std::string& path) {
		nlohmann::json json;
		{
			std::ifstream in(path);
			json = nlohmann::json::parse(in);
		}

		m_simulator->pause();

		if (m_calcCoverage) {
			m_simulator->cleanupCoverageCalculation();
		}

		m_heightMapPath = "";
		m_resistanceMapPath = "";

		// Application
		m_vSync = json["vSync"]; //
		m_calcCoverage = json["calcCoverage"]; //
		m_coverageThreshold = json["coverageThreshold"]; //
		m_targetFrameRate = json["targetFrameRate"]; //
		m_constantCoverage = json["constantCoverage"]; //
		m_constantCoverageAllowRemove = json["constantCoverageAllowRemove"]; //
		m_targetCoverage = json["targetCoverage"]; //
		m_coverageSpawnAmount = json["coverageSpawnAmount"]; //
		m_coverageSubtractAmount = json["coverageSubtractAmount"]; //
		m_coverageRadius = json["coverageRadius"]; //
		m_coverageSpawnUniform = json["coverageSpawnUniform"]; //
		m_spawnSteps = json["spawnSteps"]; //
		m_stopIterations = json["stopIterations"]; //

		// Simulation
		m_useBilinear = json["bilinear"];
		m_gridSize = { json["gridSize"][0], json["gridSize"][1] }; //
		m_gridScale = json["gridScale"]; //
		m_windAngle = json["windAngle"]; //
		m_secondWindAngle = json["secondWindAngle"]; //
		m_windBidirectionalR = json["windBidirectionalR"]; //
		m_windBidirectionalBaseTime = json["windBidirectionalBaseTime"]; //
		m_enableBidirectional = json["enableBidirectional"]; //
		m_bidirectionalStrengthBased = json["bidirectionalStrengthBased"]; //
		m_windSpeed = json["windSpeed"]; //
		m_venturiStrength = json["venturiStrength"]; //

		m_windWarpingMode = getIndexFromNamedArray(windWarpingModes, IM_ARRAYSIZE(windWarpingModes), json["windWarpingMode"], 0); //
		m_windWarpingCount = json["windWarpingCount"]; //
		m_windWarpingDivisor = json["windWarpingDivisor"]; //
		m_windWarpingRadii = json["windWarpingRadii"]; //
		m_windWarpingStrengths = json["windWarpingStrengths"]; //
		m_windWarpingGradientStrengths = json["windWarpingGradientStrengths"]; //

		m_windShadowMode = getIndexFromNamedArray(windShadowModes, IM_ARRAYSIZE(windShadowModes), json["windShadowMode"], 0); //
		m_windShadowDistance = json["windShadowDistance"]; //
		m_minWindShadowAngle = json["minWindShadowAngle"]; //
		m_maxWindShadowAngle = json["maxWindShadowAngle"]; //

		m_abrasionStrength = json["abrasionStrength"]; //
		if (json.contains("soilAbrasionStrength")) {
			m_soilAbrasionStrength = json["soilAbrasionStrength"];
		}
		m_abrasionThreshold = json["abrasionThreshold"]; //
		m_saltationMode = getIndexFromNamedArray(saltationModes, IM_ARRAYSIZE(saltationModes), json["saltationMode"], 1); //
		m_saltationStrength = json["saltationStrength"]; //
		m_reptationStrength = json["reptationStrength"]; //
		if (json.contains("reptationSmoothingStrength"))
			m_reptationSmoothingStrength = json["reptationSmoothingStrength"];
		if (json.contains("reptationUseWindShadow"))
			m_reptationUseWindShadow = json["reptationUseWindShadow"];

		m_bedrockAvalancheMode = getIndexFromNamedArray(bedrockAvalancheModes, IM_ARRAYSIZE(bedrockAvalancheModes), json["bedrockAvalancheMode"], 0); //
		m_avalancheIterations = json["avalancheIterations"]; //
		if (json.contains("pressureProjectionIterations")) {
			m_pressureProjectionIterations = json["pressureProjectionIterations"];
		}
		if (json.contains("projectionMode")) {
			m_projectionMode = json["projectionMode"];
		}
		m_bedrockAvalancheIterations = json["bedrockAvalancheIterations"]; //
		m_avalancheAngle = json["avalancheAngle"]; //
		m_bedrockAngle = json["bedrockAngle"]; //
		m_vegetationAngle = json["vegetationAngle"]; //
		if (json.contains("soilAvalancheIterations")) {
			m_soilAvalancheIterations = json["soilAvalancheIterations"];
			m_soilAngle = json["soilAngle"];
			m_vegetationSoilAngle = json["vegetationSoilAngle"];
		}

		if (json.contains("wavePeriod")) {
			m_wavePeriod = json["wavePeriod"];
			m_waveStrength = json["waveStrength"];
			m_waveDepthScale = json["waveDepthScale"];
		}

		if (json.contains("sedimentCapacityConstant")) {
			m_sedimentCapacityConstant = json["sedimentCapacityConstant"];
			m_sedimentDepositionConstant = json["sedimentDepositionConstant"];
			m_sandDissolutionConstant = json["sandDissolutionConstant"];
			m_soilDissolutionConstant = json["soilDissolutionConstant"];
			m_bedrockDissolutionConstant = json["bedrockDissolutionConstant"];
		}

		if (json.contains("sandMoistureRate")) {
			m_waterBorderLevel = json["waterBorderLevel"];
			m_waterLevel = json["waterLevel"];
			m_moistureEvaporationScale = json["moistureEvaporationScale"];
			m_sandMoistureRate = json["sandMoistureRate"];
			m_soilMoistureRate = json["soilMoistureRate"];
			m_terrainThicknessMoistureThreshold = json["terrainThicknessMoistureThreshold"];
			m_moistureCapacityConstant = json["moistureCapacityConstant"];
		}

		if (json.contains("rainStrength")) {
			m_evaporationRate = json["evaporationRate"];
			m_rainStrength = json["rainStrength"];
			m_rainPeriod = json["rainPeriod"];
			m_rainScale = json["rainScale"];
			m_rainProbabilityMin = json["rainProbabilityMin"];
			m_rainProbabilityMax = json["rainProbabilityMax"];
			m_rainProbabilityHeightRange = json["rainProbabilityHeightRange"];
		}

		m_timeMode = getIndexFromNamedArray(timeModes, IM_ARRAYSIZE(timeModes), json["timeMode"], 1); //
		m_timeScale = json["timeScale"]; //
		m_fixedDeltaTime = json["fixedDeltaTime"]; //


		const nlohmann::json& initP = json["initializationParameters"];
		for (auto& el : initP.items()) {
			int idx = getIndexFromNamedArray(initializationTargets, IM_ARRAYSIZE(initializationTargets), el.key(), -1);
			if (idx >= 0 && idx < NumNoiseGenerationTargets) {
				auto& obj = el.value();
				auto& params = m_initializationParameters.noiseGenerationParameters[idx];
				params.flat = obj["flat"];
				params.enabled = obj["enable"];
				params.uniform_random = obj["uniform_random"];
				params.iters = obj["iterations"];
				params.stretch = { obj["stretch"][0], obj["stretch"][1] };
				params.offset = { obj["offset"][0], obj["offset"][1] };
				params.border = { obj["border"][0], obj["border"][1] };
				params.scale = obj["scale"];
				params.bias = obj["bias"];
			}
		} //

		m_renderParameters.sandColor = { json["sandColor"][0], json["sandColor"][1], json["sandColor"][2], json["sandColor"][3] };
		m_renderParameters.bedrockColor = { json["bedrockColor"][0], json["bedrockColor"][1], json["bedrockColor"][2], json["bedrockColor"][3] };
		m_renderParameters.windShadowColor = { json["windShadowColor"][0], json["windShadowColor"][1], json["windShadowColor"][2], json["windShadowColor"][3] };
		m_renderParameters.vegetationColor = { json["vegetationColor"][0], json["vegetationColor"][1], json["vegetationColor"][2], json["vegetationColor"][3] };
		if (json.contains("soilColor")) {
			m_renderParameters.soilColor = { json["soilColor"][0], json["soilColor"][1], json["soilColor"][2], json["soilColor"][3] };
			m_renderParameters.humusColor = { json["humusColor"][0], json["humusColor"][1], json["humusColor"][2], json["humusColor"][3] };
			m_renderParameters.waterColor = { json["waterColor"][0], json["waterColor"][1], json["waterColor"][2], json["waterColor"][3] };
			m_renderParameters.wetColor = { json["wetColor"][0], json["wetColor"][1], json["wetColor"][2], json["wetColor"][3] };
			m_renderParameters.shadowStrength = { json["shadowStrength"] };
			render_vegetation = json["renderVegetation"];
		}

		initializeAll();

		bool exportMaps =  json.contains("exportMaps") ? json["exportMaps"].get<bool>() : false;

		if (exportMaps) {
			std::string terrainMapPath = path + ".terrain.exr";
			std::string resistanceMapPath = path + ".resistance.exr";

			if (std::filesystem::exists(terrainMapPath)) {
				if (loadEXR(m_simulator->getTerrainMap(), terrainMapPath)) {
					m_heightMapPath = terrainMapPath;
				}
			}
			if (std::filesystem::exists(resistanceMapPath)) {
				if (loadEXR(m_simulator->getResistanceMap(), resistanceMapPath)) {
					m_resistanceMapPath = resistanceMapPath;
				}
			}
		}

		if (m_calcCoverage) {
			m_simulator->setupCoverageCalculation();
		}

		return true;
	}

	bool UI::toJson(const std::string& path) {
		nlohmann::json json;

		// Application
		json["vSync"] = m_vSync;
		json["calcCoverage"] = m_calcCoverage;
		json["coverageThreshold"] = m_coverageThreshold;
		json["targetFrameRate"] = m_targetFrameRate;
		json["constantCoverage"] = m_constantCoverage;
		json["constantCoverageAllowRemove"] = m_constantCoverageAllowRemove;
		json["targetCoverage"] = m_targetCoverage;
		json["coverageSpawnAmount"] = m_coverageSpawnAmount;
		json["coverageSubtractAmount"] = m_coverageSubtractAmount;
		json["coverageRadius"] = m_coverageRadius; 
		json["coverageSpawnUniform"] = m_coverageSpawnUniform; 
		json["spawnSteps"] = m_spawnSteps;
		json["stopIterations"] = m_stopIterations;

		// Simulation
		json["bilinear"] = m_useBilinear;
		json["gridSize"] = { m_gridSize.x, m_gridSize.y };
		json["gridScale"] = m_gridScale;
		json["windAngle"] = m_windAngle;
		json["secondWindAngle"] = m_secondWindAngle;
		json["windBidirectionalR"] = m_windBidirectionalR;
		json["windBidirectionalBaseTime"] = m_windBidirectionalBaseTime;
		json["enableBidirectional"] = m_enableBidirectional;
		json["bidirectionalStrengthBased"] = m_bidirectionalStrengthBased;
		json["windSpeed"] = m_windSpeed;
		json["venturiStrength"] = m_venturiStrength;

		json["windWarpingMode"] = windWarpingModes[m_windWarpingMode];
		json["windWarpingCount"] = m_windWarpingCount;
		json["windWarpingDivisor"] = m_windWarpingDivisor;
		json["windWarpingRadii"] = m_windWarpingRadii;
		json["windWarpingStrengths"] = m_windWarpingStrengths;
		json["windWarpingGradientStrengths"] = m_windWarpingGradientStrengths; //

		json["windShadowMode"] = windShadowModes[m_windShadowMode];
		json["windShadowDistance"] = m_windShadowDistance;
		json["minWindShadowAngle"] = m_minWindShadowAngle;
		json["maxWindShadowAngle"] = m_maxWindShadowAngle;

		json["abrasionStrength"] = m_abrasionStrength;
		json["soilAbrasionStrength"] = m_soilAbrasionStrength;
		json["abrasionThreshold"] = m_abrasionThreshold;
		json["saltationMode"] = saltationModes[m_saltationMode];
		json["saltationStrength"] = m_saltationStrength;
		json["reptationStrength"] = m_reptationStrength;
		json["reptationSmoothingStrength"] = m_reptationSmoothingStrength;
		json["reptationUseWindShadow"] = m_reptationUseWindShadow;

		json["bedrockAvalancheMode"] = bedrockAvalancheModes[m_bedrockAvalancheMode];
		json["avalancheIterations"] = m_avalancheIterations;
		json["pressureProjectionIterations"] = m_pressureProjectionIterations;
		json["bedrockAvalancheIterations"] = m_bedrockAvalancheIterations;
		json["projectionMode"] = m_projectionMode;
		json["avalancheAngle"] = m_avalancheAngle;
		json["bedrockAngle"] = m_bedrockAngle;
		json["vegetationAngle"] = m_vegetationAngle;
		json["soilAvalancheIterations"] = m_soilAvalancheIterations;
		json["soilAngle"] = m_soilAngle;
		json["vegetationSoilAngle"] = m_vegetationSoilAngle;

		json["wavePeriod"] = m_wavePeriod;
		json["waveStrength"] = m_waveStrength;
		json["waveDepthScale"] = m_waveDepthScale;

		json["sedimentCapacityConstant"] = m_sedimentCapacityConstant;
		json["sedimentDepositionConstant"] = m_sedimentDepositionConstant;
		json["sandDissolutionConstant"] = m_sandDissolutionConstant;
		json["soilDissolutionConstant"] = m_soilDissolutionConstant ;
		json["bedrockDissolutionConstant"] = m_bedrockDissolutionConstant;

		json["waterBorderLevel"] = m_waterBorderLevel;
		json["waterLevel"] = m_waterLevel;
		json["moistureEvaporationScale"] = m_moistureEvaporationScale;
		json["sandMoistureRate"] = m_sandMoistureRate;
		json["soilMoistureRate"] = m_soilMoistureRate;
		json["terrainThicknessMoistureThreshold"] = m_terrainThicknessMoistureThreshold;
		json["moistureCapacityConstant"] = m_moistureCapacityConstant;

		json["evaporationRate"] = m_evaporationRate;
		json["rainStrength"] = m_rainStrength;
		json["rainPeriod"] = m_rainPeriod;
		json["rainScale"] = m_rainScale;
		json["rainProbabilityMin"] = m_rainProbabilityMin;
		json["rainProbabilityMax"] = m_rainProbabilityMax;
		json["rainProbabilityHeightRange"] = m_rainProbabilityHeightRange;

		json["timeMode"] = timeModes[m_timeMode];
		json["timeScale"] = m_timeScale;
		json["fixedDeltaTime"] = m_fixedDeltaTime;

		json["initializationParameters"] = nlohmann::json::object();
		for (int i = 0; i < NumNoiseGenerationTargets; ++i) {
			auto& params = m_initializationParameters.noiseGenerationParameters[i];
			auto obj = nlohmann::json::object();
			obj["flat"] = params.flat;
			obj["enable"] = params.enabled;
			obj["uniform_random"] = params.uniform_random;
			obj["iterations"] = params.iters;
			obj["stretch"] = { params.stretch.x, params.stretch.y };
			obj["offset"] = { params.offset.x, params.offset.y };
			obj["border"] = { params.border.x, params.border.y };
			obj["scale"] = params.scale;
			obj["bias"] = params.bias;
			json["initializationParameters"][initializationTargets[i]] = obj;
		}

		json["sandColor"] = { m_renderParameters.sandColor.x,
			m_renderParameters.sandColor.y,
			m_renderParameters.sandColor.z,
			m_renderParameters.sandColor.w
		};
		json["bedrockColor"] = { m_renderParameters.bedrockColor.x,
			m_renderParameters.bedrockColor.y,
			m_renderParameters.bedrockColor.z,
			m_renderParameters.bedrockColor.w
		};
		json["windShadowColor"] = { m_renderParameters.windShadowColor.x,
			m_renderParameters.windShadowColor.y,
			m_renderParameters.windShadowColor.z,
			m_renderParameters.windShadowColor.w
		};
		json["vegetationColor"] = { m_renderParameters.vegetationColor.x,
			m_renderParameters.vegetationColor.y,
			m_renderParameters.vegetationColor.z,
			m_renderParameters.vegetationColor.w
		};

		json["soilColor"] = { 
			m_renderParameters.soilColor.x,
			m_renderParameters.soilColor.y,
			m_renderParameters.soilColor.z,
			m_renderParameters.soilColor.w
		};

		json["humusColor"] = { 
			m_renderParameters.humusColor.x,
			m_renderParameters.humusColor.y,
			m_renderParameters.humusColor.z,
			m_renderParameters.humusColor.w
		};

		json["waterColor"] = { 
			m_renderParameters.waterColor.x,
			m_renderParameters.waterColor.y,
			m_renderParameters.waterColor.z,
			m_renderParameters.waterColor.w
		};

		json["wetColor"] = { 
			m_renderParameters.wetColor.x,
			m_renderParameters.wetColor.y,
			m_renderParameters.wetColor.z,
			m_renderParameters.wetColor.w
		};
		json["shadowStrength"] = m_renderParameters.shadowStrength;
		json["renderVegetation"] = render_vegetation;

		json["exportMaps"] = m_exportMaps;
		if (m_exportMaps) {
			std::string terrainMapPath = path + ".terrain.exr";
			std::string resistanceMapPath = path + ".resistance.exr";
			const int width = m_simulator->getTerrainMap()->getWidth();
			const int height = m_simulator->getTerrainMap()->getHeight();
			std::vector<float> data(width * height * 4);
			m_simulator->getTerrainMap()->download(data,
				width,
				height,
				GL_RGBA,
				GL_FLOAT,
				0);
			SaveEXR(data.data(), width, height, terrainMapPath.c_str(), TINYEXR_PIXELTYPE_FLOAT);
			m_simulator->getResistanceMap()->download(data,
				width,
				height,
				GL_RGBA,
				GL_FLOAT,
				0);
			SaveEXR(data.data(), width, height, resistanceMapPath.c_str(), TINYEXR_PIXELTYPE_HALF);
		}

		auto str = json.dump(1);
		std::ofstream o(path);
		o << str;
		o.close();
		return o.good();
	}

	void UI::createSceneNode()
	{
		if (ImGui::TreeNode("Scene"))
		{
			if (ImGui::Button("Reset"))
			{
				m_mean_frametime = 0.f;
				m_simulator->setInitializationParameters(m_initializationParameters);
				m_simulator->reinitialize(m_gridSize, m_gridScale);
				if (!m_heightMapPath.empty()) {
					loadEXR(m_simulator->getTerrainMap(), m_heightMapPath);
				}
				if (!m_resistanceMapPath.empty()) {
					loadEXR(m_simulator->getResistanceMap(), m_resistanceMapPath);
				}
			}

			char const* filterPatterns[1] = { "*.exr" };
			if (ImGui::Button("Load Heights from EXR")) {
				auto input = tinyfd_openFileDialog("Load Heightmap", "./", 1, filterPatterns, "OpenEXR (.exr)", 0);

				if (input != nullptr) {
					m_heightMapPath = input;
					if (!loadEXR(m_simulator->getTerrainMap(), m_heightMapPath)) {
						m_heightMapPath = "";
					}
				}
				else {
					m_heightMapPath = "";
				}
			}
			ImGui::LabelText("Selected File##height", m_heightMapPath.empty() ? "None" : m_heightMapPath.c_str());
			ImGui::SameLine();
			if (ImGui::Button("Clear##height")) {
				m_heightMapPath = "";
			}
			if (ImGui::Button("Load Resistances from EXR")) {
				auto input = tinyfd_openFileDialog("Load Resistancemap", "./", 1, filterPatterns, "OpenEXR (.exr)", 0);

				if (input != nullptr) {
					m_resistanceMapPath = input;
					if (!loadEXR(m_simulator->getResistanceMap(), m_resistanceMapPath)) {
						m_resistanceMapPath = "";
					}
				}
				else {
					m_resistanceMapPath = "";
				}
			}
			ImGui::LabelText("Selected File##resistance", m_resistanceMapPath.empty() ? "None" : m_resistanceMapPath.c_str());
			ImGui::SameLine();
			if (ImGui::Button("Clear##resistance")) {
				m_resistanceMapPath = "";
			}

			if (ImGui::Button("Save Heights to EXR")) {
				auto output = tinyfd_saveFileDialog("Save Heightmap", "./heights.exr", 1, filterPatterns, "OpenEXR (.exr)");
				if (output != nullptr) {
					const int width = m_simulator->getTerrainMap()->getWidth();
					const int height = m_simulator->getTerrainMap()->getHeight();
					std::vector<float> data(width * height * 4);
					m_simulator->getTerrainMap()->download(data,
						width,
						height,
						GL_RGBA,
						GL_FLOAT,
						0);
					SaveEXR(data.data(), width, height, output, TINYEXR_PIXELTYPE_FLOAT);
				}
			}
			if (ImGui::Button("Save Resistances to EXR")) {
				auto output = tinyfd_saveFileDialog("Save Resistancemap", "./resistances.exr", 1, filterPatterns, "OpenEXR (.exr)");
				if (output != nullptr) {
					const int width = m_simulator->getResistanceMap()->getWidth();
					const int height = m_simulator->getResistanceMap()->getHeight();
					std::vector<float> data(width * height * 4);
					m_simulator->getResistanceMap()->download(data,
						width,
						height,
						GL_RGBA,
						GL_FLOAT,
						0);
					SaveEXR(data.data(), width, height, output, TINYEXR_PIXELTYPE_HALF); // Half Precision should be enough here since all values are [0,1]
				}
			}

			for (int i = 0; i < NumNoiseGenerationTargets; ++i) {
				ImGui::PushID(i);
				if (ImGui::TreeNode(initializationTargets[i])) {
					auto& params = m_initializationParameters.noiseGenerationParameters[i];
					ImGui::Checkbox("Flat (Bias only)", &params.flat);
					ImGui::Checkbox("Uniform Random (Bias + Scale)", &params.uniform_random);
					ImGui::Checkbox("Enable", &params.enabled);
					ImGui::DragInt("Noise Iterations", &params.iters, 0.1f, 0, 50);
					ImGui::DragFloat2("Noise Stretch", &params.stretch.x, 1.f, 0.f, 100.f);
					ImGui::DragFloat2("Noise Offset", &params.offset.x, 1.f);
					ImGui::DragFloat2("Seamless Border", &params.border.x, 0.01f, 0.f, 1.f);
					ImGui::DragFloat("Height Scale", &params.scale, 1.f, 0.f, 10000.f);
					ImGui::DragFloat("Height Bias", &params.bias, 0.1f);
					ImGui::TreePop();
				}
				ImGui::PopID();
			}

			ImGui::InputInt2("Grid Size", &m_gridSize.x);
			ImGui::DragFloat("Grid Scale", &m_gridScale);

			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Save/Load JSON")) {
			ImGui::Checkbox("Export Maps to EXR", &m_exportMaps);
			if (ImGui::Button("Save")) {
				char const* filterPatterns[1] = { "*.json" };
				auto output = tinyfd_saveFileDialog("Save JSON", "./scene.json", 1, filterPatterns, "JSON (.json)");
				if (output != nullptr) {
					toJson(output);
				}
			}
			if (ImGui::Button("Load")) {
				char const* filterPatterns[1] = { "*.json" };
				auto input = tinyfd_openFileDialog("Save JSON", "./scene.json", 1, filterPatterns, "JSON (.json)", 0);
				if (input != nullptr) {
					fromJson(input);
				}
			}
			ImGui::TreePop();
		}
	}

	void UI::createSimulationNode()
	{
		if (ImGui::TreeNode("Simulation"))
		{
			if (m_simulator->isPaused())
			{
				if (ImGui::Button("Resume"))
				{
					m_simulator->resume();
				}
			}
			else
			{
				if (ImGui::Button("Pause"))
				{
					m_simulator->pause();
				}
			}

			if (ImGui::Checkbox("Use Bilinear", &m_useBilinear)) {
				m_simulator->setUseBilinear(m_useBilinear);
			}

			if (ImGui::DragInt("Stop after", &m_stopIterations, 0.1f, 0, 10000)) {
				m_simulator->setStopIterations(m_stopIterations);
			}
			ImGui::Text("Iterations: %i", m_simulator->getTimeStep());

			if (ImGui::TreeNode("Coverage")) {
				if (ImGui::Checkbox("Calculate Coverage", &m_calcCoverage)) {
					if (m_calcCoverage) {
						m_simulator->setupCoverageCalculation();
					}
					else {
						m_simulator->cleanupCoverageCalculation();
					}
				}
				if (m_calcCoverage) {
					if (ImGui::Checkbox("Constant Coverage", &m_constantCoverage)) {
						m_simulator->setConstantCoverage(m_constantCoverage);
					}
					if (ImGui::Checkbox("Allow Removal", &m_constantCoverageAllowRemove)) {
						m_simulator->setConstantCoverageAllowRemove(m_constantCoverageAllowRemove);
					}
					if (ImGui::Checkbox("Uniform", &m_coverageSpawnUniform)) {
						m_simulator->setCoverageSpawnUniform(m_coverageSpawnUniform);
					}
					if (ImGui::DragFloat("Target Coverage", &m_targetCoverage)) {
						m_simulator->setTargetCoverage(m_targetCoverage);
					}
					if (ImGui::DragFloat("Spawn Amount", &m_coverageSpawnAmount)) {
						m_simulator->setCoverageSpawnAmount(m_coverageSpawnAmount);
					}
					if (ImGui::DragFloat("Subtract Amount", &m_coverageSubtractAmount)) {
						m_simulator->setCoverageSubtractAmount(m_coverageSubtractAmount);
					}
					if (ImGui::DragInt("Spawn radius", &m_coverageRadius)) {
						m_simulator->setCoverageRadius(m_coverageRadius);
					}
					if (ImGui::DragInt("Spawn every n steps", &m_spawnSteps)) {
						m_simulator->setSpawnSteps(m_spawnSteps);
					}
				}
				if (ImGui::DragFloat("Threshold", &m_coverageThreshold, 0.0001f, 0.f, 1.f, "%.6f")) {
					m_simulator->setCoverageThreshold(m_coverageThreshold);
				}
				ImGui::Text("Coverage: %f%", m_simulator->getCoverage() * 100.f);
				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Wind")) {
				if (ImGui::Combo("Projection Mode", &m_projectionMode, projectionModes, IM_ARRAYSIZE(projectionModes)))
				{
					m_simulator->setProjectionMode(static_cast<ProjectionMode>(m_projectionMode));
				}

				if (ImGui::DragInt("Iterations", &m_pressureProjectionIterations))
				{
					m_simulator->setPressureProjectionIterations(m_pressureProjectionIterations);
				}

				if (ImGui::DragFloat("Speed", &m_windSpeed))
				{
					m_simulator->setWindSpeed(m_windSpeed);
				}
				if (ImGui::DragFloat("Angle", &m_windAngle))
				{
					m_simulator->setWindAngle(m_windAngle);
				}

				if (ImGui::DragFloat("Venturi", &m_venturiStrength, 0.005f))
				{
					m_simulator->setVenturiStrength(m_venturiStrength);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Bidirectional Wind Scheme"))
			{
				if (ImGui::Checkbox("Enable", &m_enableBidirectional))
				{
					m_simulator->enableBidirectional(m_enableBidirectional);
				}
				if (ImGui::Checkbox("Strength based", &m_bidirectionalStrengthBased))
				{
					m_simulator->setBidirectionalStrengthBased(m_bidirectionalStrengthBased);
				}
				if (ImGui::DragFloat("Second Angle", &m_secondWindAngle))
				{
					m_simulator->setSecondWindAngle(m_secondWindAngle);
				}
				if (ImGui::DragFloat("Ratio", &m_windBidirectionalR))
				{
					m_simulator->setBidirectionalR(m_windBidirectionalR);
				}
				if (ImGui::DragFloat("Period", &m_windBidirectionalBaseTime))
				{
					m_simulator->setBidirectionalBaseTime(m_windBidirectionalBaseTime);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Waves")) {
				if (ImGui::DragFloat("Wave Period", &m_wavePeriod, 0.01f)) {
					m_simulator->setWavePeriod(m_wavePeriod);
				}
				if (ImGui::DragFloat("Wave Strength", &m_waveStrength, 0.001f)) {
					m_simulator->setWaveStrength(m_waveStrength);
				}
				if (ImGui::DragFloat("Wave Depth Scale", &m_waveDepthScale, 0.01f)) {
					m_simulator->setWaveDepthScale(m_waveDepthScale);
				}
				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Vegetation"))
			{
				bool matrixUpdate = false;

				for (int i = 0; i < m_vegTypeCount; ++i)
				{
					if (ImGui::TreeNode(("Type " + std::to_string(i)).c_str()))
					{
						if (ImGui::Button("Load Mesh"))
						{
							char const* filterPatterns[2] = { "*.obj", "*.gltf" };
							auto input = tinyfd_openFileDialog("Load Mesh", m_vegMeshes[i].c_str(), 1, filterPatterns, nullptr, 0);
							if (input != nullptr) {
								m_simulator->setVegetationTypeMesh(i, std::filesystem::path{ input });
								m_vegMeshes[i] = input;
							}
						}

						ImGui::Text(m_vegMeshes[i].c_str());
				
						bool changed = ImGui::DragFloat("Max Radius", &m_vegTypes[i].maxRadius, 0.01f);
						changed |= ImGui::DragFloat("Growth Rate", &m_vegTypes[i].growthRate, 0.01f);
						changed |= ImGui::DragFloat("Position Adjust Rate", &m_vegTypes[i].positionAdjustRate, 0.01f);
						changed |= ImGui::DragFloat("Damage Rate", &m_vegTypes[i].damageRate, 0.01f);
						changed |= ImGui::DragFloat("Shrink Rate", &m_vegTypes[i].shrinkRate, 0.01f);
						changed |= ImGui::DragFloat("Max Maturity Time", &m_vegTypes[i].maxMaturityTime, 0.01f);
						changed |= ImGui::DragFloat("Maturity Percentage", &m_vegTypes[i].maturityPercentage, 0.01f);
						changed |= ImGui::DragFloat2("Height", &m_vegTypes[i].height.x, 0.01f);
						changed |= ImGui::DragFloat("Water Usage Rate", &m_vegTypes[i].waterUsageRate, 0.01f);
						changed |= ImGui::DragFloat("Water Storage Capacity", &m_vegTypes[i].waterStorageCapacity, 0.01f);
						changed |= ImGui::DragFloat("Water Resistance", &m_vegTypes[i].waterResistance, 0.01f);
						changed |= ImGui::DragFloat("Min Moisture", &m_vegTypes[i].minMoisture, 0.01f);
						changed |= ImGui::DragFloat("Max Moisture", &m_vegTypes[i].maxMoisture, 0.01f);
						changed |= ImGui::DragFloat("Soil Compatability", &m_vegTypes[i].soilCompatibility, 0.01f);
						changed |= ImGui::DragFloat("Sand Compatability", &m_vegTypes[i].sandCompatibility, 0.01f);
						changed |= ImGui::DragFloat2("Terrain Coverage Resistance", &m_vegTypes[i].terrainCoverageResistance.x, 0.01f);
						changed |= ImGui::DragFloat("Max Slope", &m_vegTypes[i].maxSlope, 0.01f);
						changed |= ImGui::DragFloat("Base Spawn Rate", &m_vegTypes[i].baseSpawnRate, 0.01f);
						changed |= ImGui::DragFloat("Density Spawn Multiplier", &m_vegTypes[i].densitySpawnMultiplier, 0.01f);
						changed |= ImGui::DragFloat("Wind Spawn Multiplier", &m_vegTypes[i].windSpawnMultiplier, 0.01f);
						changed |= ImGui::DragFloat("Humus Rate", &m_vegTypes[i].humusRate, 0.01f);
						changed |= ImGui::DragFloat2("Light Interval", &m_vegTypes[i].lightConditions.x, 0.01f);

						if (changed)
						{
							m_simulator->setVegetationType(i, m_vegTypes[i]);
						}

						if (ImGui::TreeNode("Dominance"))
						{
							for (int j = 0; j < m_vegTypeCount; ++j)
							{
								matrixUpdate |= ImGui::DragFloat(("Type " + std::to_string(j)).c_str(), & m_vegMatrix[i * c_maxVegTypeCount + j], 0.01f);
							}

							ImGui::TreePop();
						}

						ImGui::TreePop();
					}
				}

				// Add
				if (matrixUpdate)
				{
					m_simulator->setVegMatrix(m_vegMatrix);
				}

				if (ImGui::Button("Add") && m_vegTypeCount < c_maxVegTypeCount)
				{
					++m_vegTypeCount;
					int i = m_simulator->addVegType();

					m_vegMeshes[i] = std::string{};
					m_vegTypes[i] = VegetationType{};

					std::fill(m_vegMatrix.begin() + i * c_maxVegTypeCount, m_vegMatrix.begin() + (i + 1) * c_maxVegTypeCount, 1.0f);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Moisture")) {
				if (ImGui::DragFloat("Moisture Evaporation Scale", &m_moistureEvaporationScale, 0.01f)) {
					m_simulator->setMoistureEvaporationScale(m_moistureEvaporationScale);
				}
				if (ImGui::DragFloat("Sand Moisture Rate", &m_sandMoistureRate, 0.01f)) {
					m_simulator->setSandMoistureRate(m_sandMoistureRate);
				}
				if (ImGui::DragFloat("Soil Moisture Rate", &m_soilMoistureRate, 0.01f)) {
					m_simulator->setSoilMoistureRate(m_soilMoistureRate);
				}
				if (ImGui::DragFloat("Terrain Thickness Threshold", &m_terrainThicknessMoistureThreshold, 0.01f)) {
					m_simulator->setTerrainThicknessMoistureThreshold(m_terrainThicknessMoistureThreshold);
				}
				if (ImGui::DragFloat("Moisture Capacity", &m_moistureCapacityConstant, 0.01f)) {
					m_simulator->setMoistureCapacityConstant(m_moistureCapacityConstant);
				}
				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Rain")) {
				if (ImGui::DragFloat("Water Border Level", &m_waterBorderLevel, 0.1f)) {
					m_simulator->setWaterBorderLevel(m_waterBorderLevel);
				}
				if (ImGui::DragFloat("Water Level", &m_waterLevel, 0.1f)) {
					m_simulator->setWaterLevel(m_waterLevel);
				}
				if (ImGui::DragFloat("Evaporation", &m_evaporationRate, 0.001f)) {
					m_simulator->setEvaporationRate(m_evaporationRate);
				}
				if (ImGui::DragFloat("Rain Strength", &m_rainStrength, 0.1f)) {
					m_simulator->setRainStrength(m_rainStrength);
				}
				if (ImGui::DragFloat("Rain Period", &m_rainPeriod, 0.01f)) {
					m_simulator->setRainPeriod(m_rainPeriod);
				}
				if (ImGui::DragFloat("Rain Scale", &m_rainScale, 0.1f)) {
					m_simulator->setRainScale(m_rainScale);
				}
				if (ImGui::DragFloat("Rain Min Prob.", &m_rainProbabilityMin, 0.01f)) {
					m_simulator->setRainProbabilityMin(m_rainProbabilityMin);
				}
				if (ImGui::DragFloat("Rain Max Prob.", &m_rainProbabilityMax, 0.01f)) {
					m_simulator->setRainProbabilityMax(m_rainProbabilityMax);
				}
				if (ImGui::DragFloat("Rain Prob. Height Range", &m_rainProbabilityHeightRange, 1.f)) {
					m_simulator->setRainProbabilityHeightRange(m_rainProbabilityHeightRange);
				}
				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Hydraulic Erosion")) {
				if (ImGui::DragFloat("Capacity", &m_sedimentCapacityConstant, 0.001f)) {
					m_simulator->setSedimentCapacityConstant(m_sedimentCapacityConstant);
				}
				if (ImGui::DragFloat("Deposition", &m_sedimentDepositionConstant, 0.001f)) {
					m_simulator->setSedimentDepositionConstant(m_sedimentDepositionConstant);
				}
				if (ImGui::DragFloat("Sand Dissolution", &m_sandDissolutionConstant, 0.001f)) {
					m_simulator->setSandDissolutionConstant(m_sandDissolutionConstant);
				}
				if (ImGui::DragFloat("Soil Dissolution", &m_soilDissolutionConstant, 0.001f)) {
					m_simulator->setSoilDissolutionConstant(m_soilDissolutionConstant);
				}
				if (ImGui::DragFloat("Bedrock Dissolution", &m_bedrockDissolutionConstant, 0.001f)) {
					m_simulator->setBedrockDissolutionConstant(m_bedrockDissolutionConstant);
				}
				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Wind Warping"))
			{

				if (ImGui::Combo("Mode", &m_windWarpingMode, windWarpingModes, IM_ARRAYSIZE(windWarpingModes)))
				{
					m_simulator->setWindWarpingMode(static_cast<WindWarpingMode>(m_windWarpingMode));
				}

				if (ImGui::DragInt("Count", &m_windWarpingCount))
				{
					m_simulator->setWindWarpingCount(m_windWarpingCount);
				}

				if (ImGui::DragFloat("Divisor", &m_windWarpingDivisor))
				{
					m_simulator->setWindWarpingDivisor(m_windWarpingDivisor);
				}

				if (ImGui::DragFloat4("Strenghts", m_windWarpingStrengths.data(), 0.05f))
				{
					for (int i{ 0 }; i < 4; ++i)
					{
						m_simulator->setWindWarpingStrength(i, m_windWarpingStrengths[i]);
					}
				}

				if (ImGui::DragFloat4("Gradient Strenghts", m_windWarpingGradientStrengths.data(), 0.05f))
				{
					for (int i{ 0 }; i < 4; ++i)
					{
						m_simulator->setWindWarpingGradientStrength(i, m_windWarpingGradientStrengths[i]);
					}
				}

				if (ImGui::DragFloat4("Radii", m_windWarpingRadii.data()))
				{
					for (int i{ 0 }; i < 4; ++i)
					{
						m_simulator->setWindWarpingRadius(i, m_windWarpingRadii[i]);
					}
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Wind Shadow"))
			{
				if (m_simulator->isPaused() && ImGui::Button("Update##windshadow")) {
					m_simulator->updateWindShadow();
				}
				if (ImGui::Combo("Mode", &m_windShadowMode, windShadowModes, IM_ARRAYSIZE(windShadowModes)))
				{
					m_simulator->setWindShadowMode(static_cast<WindShadowMode>(m_windShadowMode));
				}

				if (ImGui::DragFloat("Distance", &m_windShadowDistance))
				{
					m_simulator->setWindShadowDistance(m_windShadowDistance);
				}

				if (ImGui::DragFloat("Min. Angle", &m_minWindShadowAngle))
				{
					m_simulator->setMinWindShadowAngle(m_minWindShadowAngle);
				}

				if (ImGui::DragFloat("Max. Angle", &m_maxWindShadowAngle))
				{
					m_simulator->setMaxWindShadowAngle(m_maxWindShadowAngle);
				}
				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Saltation"))
			{
				if (ImGui::Combo("Mode", &m_saltationMode, saltationModes, IM_ARRAYSIZE(saltationModes)))
				{
					m_simulator->setSaltationMode(static_cast<SaltationMode>(m_saltationMode));
				}

				if (ImGui::DragFloat("Strength", &m_saltationStrength, 0.05f))
				{
					m_simulator->setSaltationStrength(m_saltationStrength);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Abrasion"))
			{
				if (ImGui::DragFloat("Bedrock Strength", &m_abrasionStrength, 0.05f))
				{
					m_simulator->setAbrasionStrength(m_abrasionStrength);
				}

				if (ImGui::DragFloat("Soil Strength", &m_soilAbrasionStrength, 0.05f))
				{
					m_simulator->setSoilAbrasionStrength(m_soilAbrasionStrength);
				}

				if (ImGui::DragFloat("Threshold", &m_abrasionThreshold, 0.05f))
				{
					m_simulator->setAbrasionThreshold(m_abrasionThreshold);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Reptation"))
			{
				if (ImGui::DragFloat("Strength", &m_reptationStrength, 0.005f))
				{
					m_simulator->setReptationStrength(m_reptationStrength);
				}

				if (ImGui::Checkbox("Use wind shadow", &m_reptationUseWindShadow)) {
					m_simulator->setReptationUseWindShadow(float(m_reptationUseWindShadow));
				}

				if (ImGui::DragFloat("Smoothing Strength", &m_reptationSmoothingStrength, 0.005f))
				{
					m_simulator->setReptationSmoothingStrength(m_reptationSmoothingStrength);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Avalanching"))
			{
				if (ImGui::DragInt("Iterations", &m_avalancheIterations))
				{
					m_simulator->setAvalancheIterations(m_avalancheIterations);
				}

				if (ImGui::DragFloat("Sand Angle", &m_avalancheAngle))
				{
					m_simulator->setAvalancheAngle(m_avalancheAngle);
				}

				if (ImGui::DragFloat("Vegetation Angle", &m_vegetationAngle))
				{
					m_simulator->setVegetationAngle(m_vegetationAngle);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Soil & Bedrock Avalanching"))
			{
				if (ImGui::Combo("Mode", &m_bedrockAvalancheMode, bedrockAvalancheModes, IM_ARRAYSIZE(bedrockAvalancheModes)))
				{
					m_simulator->setBedrockAvalancheMode(static_cast<BedrockAvalancheMode>(m_bedrockAvalancheMode));
				}

				if (ImGui::DragInt("Bedrock Iterations", &m_bedrockAvalancheIterations))
				{
					m_simulator->setBedrockAvalancheIterations(m_bedrockAvalancheIterations);
				}

				if (ImGui::DragFloat("Bedrock Angle", &m_bedrockAngle))
				{
					m_simulator->setBedrockAngle(m_bedrockAngle);
				}

				if (ImGui::DragInt("Soil Iterations", &m_soilAvalancheIterations))
				{
					m_simulator->setSoilAvalancheIterations(m_soilAvalancheIterations);
				}

				if (ImGui::DragFloat("Soil Angle", &m_soilAngle))
				{
					m_simulator->setSoilAngle(m_soilAngle);
				}

				if (ImGui::DragFloat("Soil Vegetation Angle", &m_vegetationSoilAngle))
				{
					m_simulator->setVegetationSoilAngle(m_vegetationSoilAngle);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Time"))
			{
				if (ImGui::Combo("Mode", &m_timeMode, timeModes, IM_ARRAYSIZE(timeModes)))
				{
					m_simulator->setTimeMode(static_cast<TimeMode>(m_timeMode));
				}

				if (ImGui::DragFloat("Scale", &m_timeScale))
				{
					m_simulator->setTimeScale(m_timeScale);
				}

				if (ImGui::DragFloat("Fixed Delta Time", &m_fixedDeltaTime, 0.05f))
				{
					m_simulator->setFixedDeltaTime(m_fixedDeltaTime);
				}

				ImGui::TreePop();
			}

			ImGui::TreePop();
		}
	}

}

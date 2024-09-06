#version 460 core
layout(early_fragment_tests) in;
// Constants
const int FOG_MODE_NONE = 0;
const int FOG_MODE_LINEAR = 1;
const int FOG_MODE_EXPONENTIAL = 2;
const int FOG_MODE_EXPONENTIAL2 = 3;
const int LIGHT_TYPE_SPOT = 1;
const int LIGHT_TYPE_DIRECTIONAL = 2;
const float EPSILON = 1e-6f;

// Input
struct Light
{
	vec3 position;
	unsigned int type;
	vec3 color;
	float intensity;
	vec3 attenuation;
	float range;
	vec3 direction;
	float spotOuterCutOff;
	float spotInnerCutOff;
	int pad1, pad2, pad3;
};

struct Environment
{
	vec3 ambientColor;
	float ambientIntensity;
	vec3 fogColor;
	float fogDensity;
	unsigned int fogMode;
	float fogStart;
	float fogEnd;
	int lightCount;
	Light lights[16];
};

struct Water
{
	ivec2 gridSize;
	float gridScale;
	float heightScale;
	int tesselationLevel;
	bool hasHeightMap;
};

struct Fragment
{
	vec3 position;
	vec3 normal;
	vec2 uv;
};

struct RenderParameters
{
	vec4 sandColor;
	vec4 bedrockColor;
	vec4 windShadowColor;
	vec4 vegetationColor;
	vec4 soilColor;
	vec4 waterColor;
	vec4 humusColor;
	vec4 wetColor;
};

layout(std140, binding = 0) uniform PipelineBuffer
{
	float t_time;
	float t_deltaTime;
	ivec2 t_resolution;
	mat4 t_projectionMatrix;
	mat4 t_viewMatrix;
	mat4 t_inverseViewMatrix;
	mat4 t_viewProjectionMatrix;
	Environment t_environment;
	mat4 t_modelMatrix;
	mat4 t_inverseModelMatrix;
	mat4 t_modelViewMatrix;
	mat4 t_inverseModelViewMatrix;
	ivec4 t_userID;
	Water t_water;
};

layout(std140, binding = 4) uniform RenderParametersBuffer
{
	RenderParameters renderParameters;
};

layout(binding = 2) uniform sampler2D t_heightMap;

layout(binding = 8) uniform sampler2D t_windMap;
layout(binding = 9) uniform sampler2D t_resistanceMap;
layout(binding = 10) uniform sampler2D t_moistureMap;
layout(binding = 11) uniform sampler2D t_sedimentMap;


layout(early_fragment_tests) in;
in Fragment fragment;

// Output
layout(location = 0) out vec4 fragmentColor;
layout(location = 1) out vec4 fragmentDepth;

// Functionality
vec3 getAmbientColor()
{
	const vec3 ambientColor = t_environment.ambientIntensity * t_environment.ambientColor;
	return ambientColor;
}

bool intersection(in vec3 b_min, in vec3 b_max, in vec3 r_o, in vec3 inv_r_dir, out vec2 t) {
    vec3 t1 = (b_min - r_o) * inv_r_dir;
    vec3 t2 = (b_max - r_o) * inv_r_dir;

    vec3 vtmin = min(t1, t2);
    vec3 vtmax = max(t1, t2);

    t.x = max(vtmin.x, max(vtmin.y, vtmin.z));
    t.y = min(vtmax.x, min(vtmax.y, vtmax.z));

    return t.y >= t.x;// && t.x >= 0;
}

void main()
{



	const vec3 viewVector = t_inverseViewMatrix[3].xyz - fragment.position;
	const float viewDistance = length(viewVector);
	const vec3 viewDirection = viewVector / (viewDistance + EPSILON);

	const vec4 bmin = t_modelMatrix * t_water.gridScale * vec4(0.f, -100000.f, 0.f, 1.f);
	const vec4 bmax = t_modelMatrix * t_water.gridScale * vec4(t_water.gridSize.x, 100000.f, t_water.gridSize.y, 1.f);
	vec2 t = vec2(0.f);
	t = intersection(bmin.xyz, bmax.xyz, t_inverseViewMatrix[3].xyz, -1.f/viewDirection, t) ? t : vec2(0);

	fragmentDepth = vec4(viewDistance, t.y, 0.f, 0.f);

	vec3 normal = normalize(fragment.normal);

	const vec4 terrain = texture(t_heightMap, fragment.uv);

	fragmentColor.rgb = vec3(0);

	for (int i = 0; i < t_environment.lightCount; ++i)
	{
		vec3 lightDirection;
		float attenuation;

		if (t_environment.lights[i].type == LIGHT_TYPE_DIRECTIONAL)
		{
			lightDirection = -t_environment.lights[i].direction;
			attenuation = 1.0f;
		}
		else
		{
			const vec3 lightVector = t_environment.lights[i].position - fragment.position;
			const float lightDistance2 = dot(lightVector, lightVector);
			const float lightDistance = sqrt(lightDistance2);

			if (lightDistance >= t_environment.lights[i].range)
			{
				continue;
			}

			lightDirection = lightVector / (lightDistance + EPSILON);
			attenuation = clamp(1.0f / (t_environment.lights[i].attenuation.x +
				                        t_environment.lights[i].attenuation.y * lightDistance +
				                        t_environment.lights[i].attenuation.z * lightDistance2), 0.0f, 1.0f);

			if (t_environment.lights[i].type == LIGHT_TYPE_SPOT)
			{
			    const float cosTheta = dot(lightDirection, -t_environment.lights[i].direction);

				if (cosTheta < t_environment.lights[i].spotOuterCutOff)
				{
				    continue;
				}

				attenuation *= clamp((cosTheta - t_environment.lights[i].spotOuterCutOff) /
					                 (t_environment.lights[i].spotInnerCutOff - t_environment.lights[i].spotOuterCutOff), 0.0f, 1.0f);
			}
		}

		const vec3 reflection = reflect(-lightDirection, normal);
		const float cosPsi = max(dot(reflection, viewDirection), 0.0f);
		const float cosPsiN = pow(cosPsi, 1600.0f);

		fragmentColor.rgb += attenuation * t_environment.lights[i].intensity * t_environment.lights[i].color * cosPsiN;


	}

	const float specularFactor = clamp(5.f * terrain.w, 0.f, 1.f);
	const vec3 backgroundColor = vec3(0.7f, 0.9f, 1.0f);
	const float cosTheta = max(dot(normal, viewDirection), 0.0f);
	const float cosTheta5 = pow(1 - cosTheta, 5.f);
	const float rTheta = specularFactor * (0.04 + 0.96 * cosTheta5);

	fragmentColor.rgb = specularFactor * fragmentColor.rgb / (rTheta + EPSILON) + backgroundColor;
	fragmentColor.a = rTheta;
}

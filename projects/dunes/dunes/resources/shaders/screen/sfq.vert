#version 460 core

// Output
struct Fragment
{
	vec2 uv;
};

out Fragment fragment;

void main()
{
    float x     = -1.0 + float((gl_VertexID & 1) << 2);
    float y     = -1.0 + float((gl_VertexID & 2) << 1);
    fragment.uv.x        = (x + 1.0) * 0.5;
    fragment.uv.y        = (y + 1.0) * 0.5;
    gl_Position = vec4(x, y, 0, 1);
}
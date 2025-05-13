# Oasis
This repository contains the full source code of our paper **Oasis: A Real-Time Hydraulic and Aeolian Erosion Simulation with
 Dynamic Vegetation**, which will be presented at GRAPP 2025.

https://github.com/user-attachments/assets/995d4b1e-b27d-4846-a6c9-8a517d152049

https://github.com/user-attachments/assets/57396c54-6e3b-40d8-8c65-cbd7b7a0eb2d

https://github.com/user-attachments/assets/3a79c1f7-232e-4fb7-a307-f778f4fd07d7

https://github.com/user-attachments/assets/56712edb-5e87-4903-8a9f-ae668c7a1bce

https://github.com/user-attachments/assets/364b629e-28f5-4334-91d2-f0b4f4f59fa5

# Setup
We provided a visual studio solution file. We have only tested our code on Windows, but it should be possible to get it to work on Linux. Mac might be an issue, as we are using OpenGL 4.6 for visualization.
## Requirements
The project is setup using Visual Studio 2022, CUDA 12.5 and `compute_75,sm75` flags. Older architectures and CUDA versions will work too, but you'll have to manually fix the visual studio solution to make it work.

If you are running the executable on a Laptop, make sure that it runs on your dedicated NVIDIA graphics card instead of i.e. the Intel integrated GPU. While not a problem for CUDA, the OpenGL part of our application will likely not work properly with these GPUs. Additionally,
we are sharing OpenGL textures which CUDA, which could cause additional problems if OpenGL runs on a different GPU compared to CUDA kernels.

A `vcpkg.json` manifest file has been provided. Default installs of Visual Studio 2022 already include VCPKG and should install the required packages automatically.

The list of required packages for vcpkg is as follows (in case that manual installation is needed/required):
`assimp,entt,stb,glm,glfw3,glad,imgui[glfw-binding,opengl3-binding],tinyexr,tinyfiledialogs,nlohmann-json`

# Code
The relevant simulation code is in `projects/dunes`. We further provide the scenes used in our paper in the `scenes` folder, which can be loaded with the application UI.

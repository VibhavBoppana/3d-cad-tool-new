# 🧊 3D Modeller in Python

A lightweight 3D modelling tool built in Python using Legacy OpenGL and GLUT. This application demonstrates scene graphs, hierarchical rendering, interactive controls, and object manipulation—all in under 500 lines of code.


---

## 🚀 Features

- ✅ Object-oriented scene graph with support for primitives and hierarchical nodes
- ✅ Render with Legacy OpenGL (via PyOpenGL)
- ✅ Real-time object placement, movement, scaling, and color cycling
- ✅ Ray-based object selection (picking) using AABB approximation
- ✅ Built-in primitives: Cube, Sphere, and a composite Snowman figure
- ✅ Simple keyboard and mouse interactions

---

## 📦 Requirements

- Python 3.6+
- [PyOpenGL](https://pypi.org/project/PyOpenGL/)
- [PyOpenGL_accelerate](https://pypi.org/project/PyOpenGL-accelerate/)
- FreeGLUT installed on your system
- `trackball.py` (included in this repo)

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/3d-modeller.git
cd 3d-modeller

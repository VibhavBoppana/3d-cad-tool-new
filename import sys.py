import sys
import random
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import trackball

# Constants for colors
decolor = [
    (1.0, 0.0, 0.0), # red
    (0.0, 1.0, 0.0), # green
    (0.0, 0.0, 1.0), # blue
    (1.0, 1.0, 0.0), # yellow
    (1.0, 0.0, 1.0), # magenta
    (0.0, 1.0, 1.0), # cyan
]
MIN_COLOR, MAX_COLOR = 0, len(decolor)-1

# AABB for simple intersection tests
class AABB:
    def __init__(self, min_corner, max_corner):
        self.min = np.array(min_corner, dtype=float)
        self.max = np.array(max_corner, dtype=float)

    def scale(self, s):
        center = (self.min + self.max)/2
        half = (self.max - self.min)/2 * s
        self.min = center - half
        self.max = center + half

    def ray_hit(self, start, direction, mat):
        # Transform ray into object space
        inv_mat = np.linalg.inv(mat)
        local_start = inv_mat.dot(np.append(start,1.0))[:3]
        local_dir = inv_mat.dot(np.append(direction,0.0))[:3]
        tmin, tmax = 0.0, float('inf')
        for i in range(3):
            if abs(local_dir[i]) < 1e-8:
                if local_start[i] < self.min[i] or local_start[i] > self.max[i]: return (False, None)
            else:
                t1 = (self.min[i] - local_start[i]) / local_dir[i]
                t2 = (self.max[i] - local_start[i]) / local_dir[i]
                tmin, tmax = max(tmin, min(t1,t2)), min(tmax, max(t1,t2))
                if tmin > tmax: return (False, None)
        return (True, tmin)

# Utility functions

def translation(displacement):
    t = np.identity(4)
    t[:3,3] = displacement
    return t


def scaling(scale):
    s = np.identity(4)
    s[0,0], s[1,1], s[2,2] = scale
    return s

# Base node
class Node:
    def __init__(self):
        self.color_index = random.randint(MIN_COLOR, MAX_COLOR)
        self.aabb = AABB([-0.5,-0.5,-0.5],[0.5,0.5,0.5])
        self.translation_matrix = np.identity(4)
        self.scaling_matrix = np.identity(4)
        self.selected = False
        self.depth = 0
        self.selected_loc = np.zeros(3)

    def render(self):
        glPushMatrix()
        glMultMatrixf(self.translation_matrix.T)
        glMultMatrixf(self.scaling_matrix.T)
        glColor3f(*decolor[self.color_index])
        if self.selected:
            glMaterialfv(GL_FRONT, GL_EMISSION, [0.3,0.3,0.3,1.0])
        self.render_self()
        if self.selected:
            glMaterialfv(GL_FRONT, GL_EMISSION, [0.0,0.0,0.0,1.0])
        glPopMatrix()

    def render_self(self):
        raise NotImplementedError

    def pick(self, start, direction, mat):
        newmat = mat.dot(self.translation_matrix).dot(np.linalg.inv(self.scaling_matrix))
        hit, dist = self.aabb.ray_hit(start, direction, newmat)
        return hit, dist

    def select(self, sel=None):
        self.selected = sel if sel is not None else not self.selected

    def rotate_color(self, forwards):
        self.color_index = (self.color_index + (1 if forwards else -1))
        self.color_index = MIN_COLOR + (self.color_index - MIN_COLOR) % (MAX_COLOR-MIN_COLOR+1)

    def scale(self, up):
        factor = 1.1 if up else 0.9
        self.scaling_matrix = self.scaling_matrix.dot(scaling([factor]*3))
        self.aabb.scale(factor)

    def translate(self, x, y, z):
        self.translation_matrix = self.translation_matrix.dot(translation([x,y,z]))

# Primitive shapes
class Primitive(Node):
    def __init__(self, call_list):
        super().__init__()
        self.call_list = call_list

    def render_self(self):
        glCallList(self.call_list)

class Cube(Primitive):
    def __init__(self):
        super().__init__(G_OBJ_CUBE)

class Sphere(Primitive):
    def __init__(self):
        super().__init__(G_OBJ_SPHERE)

# Hierarchical node for composites
class HierarchicalNode(Node):
    def __init__(self):
        super().__init__()
        self.child_nodes = []

    def render_self(self):
        for child in self.child_nodes:
            child.render()

class SnowFigure(HierarchicalNode):
    def __init__(self):
        super().__init__()
        s1, s2, s3 = Sphere(), Sphere(), Sphere()
        s1.translate(0, -0.6, 0)
        s2.translate(0, 0.1, 0); s2.scaling_matrix = scaling([0.8]*3)
        s3.translate(0, 0.75, 0); s3.scaling_matrix = scaling([0.7]*3)
        for c in (s1,s2,s3):
            c.color_index = MIN_COLOR
        self.child_nodes = [s1,s2,s3]
        self.aabb = AABB([0.0,0.0,0.0],[0.5,1.1,0.5])

# Scene management\class Scene:
    PLACE_DEPTH = 15.0

    def __init__(self):
        self.node_list = []
        self.selected_node = None

    def add_node(self, node):
        self.node_list.append(node)

    def render(self):
        for node in self.node_list:
            node.render()

    def pick(self, start, direction, mat):
        if self.selected_node:
            self.selected_node.select(False)
            self.selected_node = None
        mind, closest = sys.maxsize, None
        for node in self.node_list:
            hit, dist = node.pick(start,direction,mat)
            if hit and dist < mind:
                mind, closest = dist, node
        if closest:
            closest.select(True)
            closest.depth = mind
            # compute selected_loc in world space
            closest.selected_loc = start + direction*mind
            self.selected_node = closest

    def move_selected(self, start, direction, inv_mat):
        if not self.selected_node: return
        node = self.selected_node
        newloc = start + direction*node.depth
        delta = newloc - node.selected_loc
        tvec = inv_mat.dot(np.append(delta,0))[:3]
        node.translate(*tvec)
        node.selected_loc = newloc

    def rotate_selected_color(self, forwards):
        if self.selected_node: self.selected_node.rotate_color(forwards)

    def scale_selected(self, up):
        if self.selected_node: self.selected_node.scale(up)

    def place(self, shape, start, direction, inv_mat):
        newnode = {'cube':Cube,'sphere':Sphere,'figure':SnowFigure}[shape]()
        self.add_node(newnode)
        trans = start + direction*self.PLACE_DEPTH
        tvec = inv_mat.dot(np.append(trans,1.0))[:3]
        newnode.translate(*tvec)

# Initialize OpenGL call lists for primitives
G_OBJ_CUBE, G_OBJ_SPHERE = None, None
def init_primitives():
    global G_OBJ_CUBE, G_OBJ_SPHERE
    G_OBJ_CUBE = glGenLists(1)
    glNewList(G_OBJ_CUBE, GL_COMPILE)
    glutSolidCube(1.0)
    glEndList()
    G_OBJ_SPHERE = glGenLists(1)
    glNewList(G_OBJ_SPHERE, GL_COMPILE)
    glutSolidSphere(0.5,16,16)
    glEndList()

# Interaction and callbacks
from collections import defaultdict
class Interaction:
    def __init__(self, scene):
        self.scene = scene
        self.pressed = None
        self.translation = [0,0,0]
        self.trackball = trackball.Trackball(theta=-25, distance=15)
        self.mouse_loc = None
        self.callbacks = defaultdict(list)
        self.register_callbacks()

    def register_callbacks(self):
        glutMouseFunc(self.on_mouse)
        glutMotionFunc(self.on_motion)
        glutKeyboardFunc(self.on_key)
        glutSpecialFunc(self.on_key)

    def on_mouse(self, button, state, x, y):
        h = glutGet(GLUT_WINDOW_HEIGHT)
        y = h - y
        if state==GLUT_DOWN:
            self.pressed = button
            if button==GLUT_LEFT_BUTTON: self.trigger('pick',x,y)
            elif button==3: self.translate(0,0,1)
            elif button==4: self.translate(0,0,-1)
        else:
            self.pressed = None
        self.mouse_loc=(x,y)
        glutPostRedisplay()

    def on_motion(self, x, y):
        h = glutGet(GLUT_WINDOW_HEIGHT)
        y = h - y
        if self.pressed:
            dx,dy = x-self.mouse_loc[0], y-self.mouse_loc[1]
            if self.pressed==GLUT_RIGHT_BUTTON:
                self.trackball.drag_to(self.mouse_loc[0],self.mouse_loc[1],dx,dy)
            elif self.pressed==GLUT_LEFT_BUTTON:
                self.trigger('move',x,y)
            elif self.pressed==GLUT_MIDDLE_BUTTON:
                self.translate(dx/60.0,dy/60.0,0)
            glutPostRedisplay()
        self.mouse_loc=(x,y)

    def on_key(self, key, x, y):
        if key==b's': self.trigger('place','sphere',x,y)
        elif key==b'c': self.trigger('place','cube',x,y)
        glutPostRedisplay()

    def translate(self, x,y,z):
        self.translation[0]+=x; self.translation[1]+=y; self.translation[2]+=z

    def register(self,name,func):
        self.callbacks[name].append(func)

    def trigger(self,name,*a):
        for f in self.callbacks[name]: f(*a)

# Main viewer class
class Viewer:
    def __init__(self):
        glutInit()
        glutInitDisplayMode(GLUT_SINGLE|GLUT_RGB|GLUT_DEPTH)
        glutInitWindowSize(800,600)
        glutCreateWindow(b"3D Modeller")
        self.scene = Scene()
        self.inter = Interaction(self.scene)
        self.inter.register('pick', self.pick)
        self.inter.register('move', self.move)
        self.inter.register('place', self.place)
        self.inter.register('rotate_color', self.color)
        self.inter.register('scale', self.scale)
        init_primitives()
        self.init_gl()
        self.init_scene()
        glutDisplayFunc(self.render)

    def init_gl(self):
        glEnable(GL_CULL_FACE); glCullFace(GL_BACK)
        glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0,GL_POSITION,[0,0,1,0])
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE)
        glClearColor(0.4,0.4,0.4,1)

    def init_scene(self):
        c = Cube(); c.translate(2,0,2); self.scene.add_node(c)
        s = Sphere(); s.translate(-2,0,2); self.scene.add_node(s)
        f = SnowFigure(); f.translate(-2,0,-2); self.scene.add_node(f)

    def init_view(self):
        w,h = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
        glViewport(0,0,w,h)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(70, w/float(h), 0.1, 1000)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glTranslatef(*self.inter.translation)

    def render(self):
        self.init_view()
        glEnable(GL_LIGHTING); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glMultMatrixf(self.inter.trackball.matrix)
        self.scene.render()
        glDisable(GL_LIGHTING)
        glutSolidTeapot(1.0)  # sample object
        glPopMatrix()
        glFlush()

    def get_ray(self,x,y):
        self.init_view()
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        start = np.array(gluUnProject(x,y,0.001))
        end = np.array(gluUnProject(x,y,0.999))
        dir = end-start; dir/=np.linalg.norm(dir)
        return start,dir

    def pick(self,x,y):
        s,d = self.get_ray(x,y); self.scene.pick(s,d, np.linalg.inv(glGetFloatv(GL_MODELVIEW_MATRIX).T))

    def move(self,x,y):
        s,d = self.get_ray(x,y)
        inv = np.linalg.inv(glGetFloatv(GL_MODELVIEW_MATRIX).T)
        self.scene.move_selected(s,d,inv)

    def place(self,shape,x,y):
        s,d = self.get_ray(x,y)
        inv = np.linalg.inv(glGetFloatv(GL_MODELVIEW_MATRIX).T)
        self.scene.place(shape,s,d,inv)

    def color(self,forward): self.scene.rotate_selected_color(forward)
    def scale(self,up): self.scene.scale_selected(up)

    def main_loop(self):
        glutMainLoop()

if __name__ == '__main__':
    Viewer().main_loop()

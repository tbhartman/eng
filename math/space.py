""" Copyright (c) 2010, Tim Hartman

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Authored by: Tim Hartman"""

import scipy
import numpy
import copy

class Vector(numpy.ndarray):
    
    cs = None
    
    def __new__(cls, *args):
        r = numpy.ndarray([3])
        r.fill(numpy.NaN)
        r = r.view(cls)
        return r
    
    def __init__(self, x, y, z, cs=None):
        self.cs = cs
        if x is not None:
            self[0] = x
        if y is not None:
            self[1] = y
        if z is not None:
            self[2] = z
    
    def __repr__(self):
        str = numpy.ndarray.__repr__(self)
        return str.replace("\n","\n ")
    
    def unit_vector(self):
        return self / self.norm()
        
    def norm(self):
        sum = numpy.power(self,2).sum()
        return numpy.sqrt(sum)
    
    def to_cs(self, cs, pos=False):
        if pos:
            return self.to_base() - cs.to_base().origin
        else:
            return numpy.dot(self, numpy.linalg.inv(cs.to_base()))
    
    def to_base(self, pos=False):
        if pos:
            return self.cs.to_base().origin + Vector.to_base(self)
        else:
            return numpy.dot(self, self.cs.to_base())
    
    def __deepcopy__(self, memo):
        x = copy.deepcopy(self[0])
        y = copy.deepcopy(self[1])
        z = copy.deepcopy(self[2])
        cs = copy.deepcopy(self.cs)
        return Vector(x,y,z,cs)
    

class CS(numpy.ndarray):
    """XYZ Coordinate System
    
    #TODO
    For now, this is just your regular x-y-z system...perhaps after tensors I'll
    figure out how create a more general class"""
    
    #TODO parent_cs is not yet impletmented...so all CSs are referenced to main
    #CS
    parent_cs = None
    origin = None
    
    def __get_e1(self):
        return self[0].view(Vector)
    def __get_e2(self):
        return self[1].view(Vector)
    def __get_e3(self):
        return self[2].view(Vector)
    
    e1 = property(__get_e1)
    e2 = property(__get_e2)
    e3 = property(__get_e3)
    
    def __new__(cls,*args):
        r = numpy.ndarray([3,3])
        r.fill(numpy.NaN)
        r = r.view(cls)
        return r
    
    def __init__(self, origin, point_x, point_xy, parent_cs=None):
        """Initialize Coordinate System
        
        arguments:
            origin: origin of CS (Vector)
            point_x: point on x-axis (Vector)
            point_xy: point in xy-plane (Vector)"""
        
        #TODO add checks for orthogonality
        self.origin = origin
        e1 = (point_x - self.origin).unit_vector()
        vector_in_xy = (point_xy - self.origin)
        e3 = numpy.cross(e1, vector_in_xy).view(Vector).unit_vector()
        e2 = numpy.cross(e3, e1).view(Vector).unit_vector()
        
        self[0] = e1
        self[1] = e2
        self[2] = e3
        
        self.parent_cs = parent_cs
    
    def __repr__(self):
        str = numpy.ndarray.__repr__(self)
        str = str.replace("\n   ","\n")
        str += "\n" + self.origin.__repr__().replace("Vector","  ")
        return str
    
    def transform(self, transform):
        """Apply a transformation to this CS."""
        current_cs = self[:,:]
        self[:,:] = numpy.dot(self, numpy.linalg.inv(transform))
    
    def translate(self, translation):
        self.origin += translation
    
    def to_base(self):
        cs_iter = copy.deepcopy(self)
        pcs = copy.deepcopy(self.parent_cs)
        count = 0
        while pcs is not None:
            count += 1
            cs_iter[:,:] = numpy.dot(cs_iter, pcs)
            cs_iter.origin += pcs.origin
            pcs = copy.deepcopy(pcs.parent_cs)
        return cs_iter
    
    def __copy__(self):
        return CS(self.origin, self.origin + self.e1, self.origin + self.e2,
                  self.parent_cs)
    
    def __deepcopy__(self, memo):
        #TODO there has to be a better way?
        org = copy.deepcopy(self.origin)
        x = org + copy.deepcopy(self.e1)
        xy = org + copy.deepcopy(self.e2)
        cs = copy.deepcopy(self.parent_cs)
        r = CS(org, x, xy, cs)
        return r
    
    def create_child(self):
        return CS(Vector(0,0,0), Vector(1,0,0), Vector(0,1,0), self)
    
    @staticmethod
    def default():
        org = Vector(0,0,0)
        x = Vector(1,0,0)
        xy = Vector(1,1,0)
        cs = CS(org, x, xy)
        return cs
    

class Transformation(numpy.ndarray):
    
    def __new__(cls,*args):
        r = numpy.eye(3).view(Transformation)
        return r
    
    def __init__(self):
        pass
    
    def __repr__(self):
        str = numpy.ndarray.__repr__(self)
        return str.replace("\n","\n         ")
    
    @staticmethod
    def rotation(theta, axis):
        """rotation about arbitrary axis
        this is taken from .as code, not sure if its correct
        (might rotate negatively)"""
        
        # using a rotation formula:
        # http://mathworld.wolfram.com/RotationFormula.html
        # r' = O-N + N-V + V-Q
        
        #TODO confirm code
        
        nx = axis[0]
        ny = axis[1]
        nz = axis[2]
        
        p = theta
        
        c = numpy.cos(p)
        s = numpy.sin(p)
        
        rot_mat = Transformation()
        
        rot_mat[0,0] = nx * nx * (1-c) + c
        rot_mat[0,1] = nx * ny * (1-c) + nz * s
        rot_mat[0,2] = nx * nz * (1-c) - ny * s
        rot_mat[1,0] = nx * ny * (1-c) - nz * s
        rot_mat[1,1] = ny * ny * (1-c) + c
        rot_mat[1,2] = ny * nz * (1-c) + nx * s
        rot_mat[2,0] = nx * nz * (1-c) + ny * s
        rot_mat[2,1] = ny * nz * (1-c) - nx * s
        rot_mat[2,2] = nz * nz * (1-c) + c
        
        return numpy.linalg.inv(rot_mat)


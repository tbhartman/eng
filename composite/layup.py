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

import numpy
import scipy
from eng.material import elastic
from eng.math import space
import copy
from matplotlib import pyplot

class Ply(object):
    """Ply of material.

    Longer class information for Ply.  Need to decide what goes here.

    Attributes:
        theta: material rotation (in radians).
    """
    
    mat = None
    theta = 0
    thickness = 0
    z0 = 0
    
    def __get_theta_deg(self):
        return scipy.degrees(self.theta)
    def __set_theta_deg(self, value):
        self.theta = scipy.radians(value)
    theta_deg = property(__get_theta_deg, __set_theta_deg)
    
    def __init__(self, theta, thickness, mat):
        self.theta = theta
        self.thickness = thickness
        self.mat = mat
    
    def __deepcopy__(self):
        mat = copy.deepcopy(self.mat)
        theta = self.theta
        thickness = self.theata
        
        return Ply(theta, thickness, mat)
        
    

class Laminate(list):
    
    cs = space.CS.default()
    _midplane = 0 #this is the ply number (float) of z=0...0 is bottom surface
    _midplane_isset = False #True if user set
    
    def __get_midplane(self):
        return self._midplane
    def __set_midplane(self, value):
        self._midplane = value
        self._midplane_isset = True
    
    midplane = property(__get_midplane, __set_midplane)
    
    def __get_sequence(self):
        t = []
        for i in range(self.ply_count):
            t.append(self[i].theta_deg)
        #TODO i'd like to make this readable, like "[+30/-30/0/90]2s"
        return t
    sequence = property(__get_sequence)
    def __get_ply_count(self):
        return self.__len__()
    ply_count = property(__get_ply_count)
    def __get_thickness(self):
        t = 0
        for i in iter(self):
            t += i.thickness
        return t
    thickness = property(__get_thickness)
    
    def __init__(self):
        pass
    
    def append(self, item):
        index = len(self)
        self.insert(index, item)
    
    def insert(self, index, item):
        if issubclass(type(item),Ply):
            list.insert(self, index, item)
            if not self._midplane_isset:
                self.reset_midplane()
        else:
            raise TypeError('%s is not a child of %s' % (item,Ply))
    
    def prepend(self, item):
        self.insert(0, item)
    
    def __repr__(self):
        str = list.__repr__(self)
        str = str.replace(",","\n")
        return str
    
    def reset_midplane(self):
        self._midplane = self.ply_count/2.0
        self._midplane_isset = False
    
    def _get_midplane_distance_from_bottom(self):
        floor_midplane = int(numpy.floor(self.midplane))
        pos_midplane = 0
        for i in xrange(floor_midplane):
            pos_midplane += self[i].thickness
        pos_midplane += (self[floor_midplane].thickness *
                         (self.midplane - floor_midplane))
        return pos_midplane
    
    def convert_index_to_position(self, index):
        floor = int(numpy.floor(index))
        pos = 0
        for i in xrange(floor):
            pos += self[i].thickness
        if floor != self.ply_count:
            pos += self[floor].thickness * (index - floor)
        
        return pos - self._get_midplane_distance_from_bottom()
    
    def convert_position_to_index(self, pos):
        pos_from_bottom = self._get_midplane_distance_from_bottom() + pos
        index = 0
        current_pos = 0
        for i in xrange(self.ply_count):
            current_pos += self[i].thickness
            if pos_from_bottom < current_pos:
                index += 1 - (current_pos - pos_from_bottom)/self[i].thickness
                break
            else:
                index += 1
        if index < 0:
            raise IndexError('Position is out of range')
        return index
    
    def linspace(self, num, eps=5):
        # seek to do something like numpy.linspace
        num_per_ply = num / self.ply_count + 1
        lin = numpy.zeros([num_per_ply * self.ply_count])
        for i in xrange(self.ply_count):
            bot = self.convert_index_to_position(i)
            top = self.convert_index_to_position(i+1)
            start = i * num_per_ply
            stop = start + (num_per_ply - 1)
            lin[start:stop+1] = numpy.linspace(bot,top,num=num_per_ply)
            delta = ((top - bot) / num_per_ply) * 10**(-eps)
            if i != 0:
                lin[start] += delta
            if i != self.ply_count - 1:
                lin[stop] -= delta
        return lin

def format_plot_ticks(axis, lam):
    ticks = []
    ticks_minor = []
    ticks_minor_labels = []
    H = lam.thickness
    for i in xrange(lam.ply_count + 1):
        ticks.append(lam.convert_index_to_position(i) / H)
        if i < lam.ply_count:
            ticks_minor.append(lam.convert_index_to_position(i+0.5) / H)
            ticks_minor_labels.append(r'%0.1f$^\circ$' % lam[i].theta_deg)
    bot = lam.convert_index_to_position(0)
    top = lam.convert_index_to_position(lam.ply_count)
    view_interval = [min(ticks),max(ticks)]
    axis.limit_range_for_scale(*view_interval)
    axis.set_ticks(ticks)
    ticks_old = axis.get_ticklocs()
    axis.set_ticks(ticks_minor, minor=True)
    axis.set_tick_params(which='minor', width=0)
    axis.set_ticklabels(ticks_minor_labels, minor=True,
                        color='g', style='italic', size='small')
    axis.limit_range_for_scale(*view_interval)
    axis.set_ticks(ticks)
    axis.set_ticklabels(ticks_old)
    

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

import copy
import numpy as np
from eng import material
from eng.math import space

class CoefficientOfExpansion(object):
    
    _values = None
    names = None
    
    def __get_order(self):
        return self._values.shape[1]
    order = property(__get_order)
    
    def __init__(self, order=6):
        self._values = np.ndarray([0,order])
        self.names = []
    
    def __getitem__(self, index):
        if len(index) > 2 | len(index) < 1:
            #todo:raise exception
            pass
        elif len(index) == 2:
            return self._values[self.names.index(index[0]),index[1]]
        else:
            return self._values[self.names.index(index):1,:].T
    
    def __setitem__(self, index, value):
        if len(index) > 2 | len(index) < 1:
            #todo:raise exception
            pass
        else:
            if index[0] not in self.names:
                self.names.append(index[0])
                s = self._values.shape
                old = np.ndarray([s[0], s[1]])
                old[:,:] = self._values[:,:]
                self._values = np.ndarray([s[0]+1, s[1]])
                self._values[0:s[0],:] = old[:,:]
                self._values[s[0],:].fill(np.NaN)
            if len(index) == 2:
                self._values[self.names.index(index[0]),index[1]] = value
    
    def __repr__(self):
        str = repr(self._values)
        name_len_max = 0
        for i in iter(self.names):
            curr_len = len(i)
            if curr_len > name_len_max:
                name_len_max = curr_len
        fmt = '%%-%0.0fs : ' % name_len_max
        str = str.replace("array([",fmt)
        str = str.replace("],\n       ","]\n"+fmt)
        str = str.replace("])","")
        for i in iter(self.names):
            replacement = fmt % i
            str = str.replace(fmt,replacement,1)
        return str
    
    def __iter__(self):
        return iter(self.names)

class FullElasticTensor(np.ndarray):
    """Elastic fourth-order tensor
    
    """
    def __new__(cls):
        r = np.ndarray([3,3,3,3])
        r.fill(np.NaN)
        return r.view(cls)
    
    def reduced(self):
        return ConvertFullToReduced(self)
    
    def rotate(self, angle, axis):
        #trx = space.Transformation().rotation(angle, axis).transpose()
        
        #order = (1,0,3,2)
        #rotd = np.tensordot(trx,
                #np.tensordot(trx,
                 #np.tensordot(trx,
                  #np.tensordot(trx,self
                  #,[[1],[0]])
                 #,[[1],[1]])
                #,[[1],[2]])
               #,[[1],[3]])
        #new = np.empty_like(self)
        #new[:] = rotd[:]
        reduced = self.reduced().rotate(theta, axis)
        return reduced.full()
        
    
    def __setitem__(self,*args):
        #TODO tbh force symmetry
        np.ndarray.__setitem__(self,args[0],args[1])

class ReducedElasticMatrix(np.ndarray):
    """Elastic reduced matrix
    
    """
    def __new__(cls):
        r = np.ndarray([6,6])
        r.fill(np.NaN)
        return r.view(cls)
    
    def full(self):
        return ConvertReducedToFull(self)
    
    def rotate(self, theta, axis):
        #full = self.full().rotate(*args)
        m = np.cos(theta)
        n = np.sin(theta)
        t1 = np.array([[m**2,n**2,0,0,0,2*m*n],
                       [n**2,m**2,0,0,0,-2*m*n],
                       [0,0,1,0,0,0],
                       [0,0,0,m,-n,0],
                       [0,0,0,n,m,0],
                       [-m*n,m*n,0,0,0,m**2-n**2]])
        t2 = np.array([[m**2,n**2,0,0,0,m*n],
                       [n**2,m**2,0,0,0,-m*n],
                       [0,0,1,0,0,0],
                       [0,0,0,m,-n,0],
                       [0,0,0,n,m,0],
                       [-2*m*n,2*m*n,0,0,0,m**2-n**2]])
        reduced = np.dot(np.dot(np.linalg.inv(t1),self),t2)
        ret = ReducedElasticMatrix()
        ret[:] = reduced[:]
        return ret
        
    def __setitem__(self,*args):
        #TODO tbh force symmetry
        np.ndarray.__setitem__(self,args[0],args[1])
    
    def __repr__(self, prec=2):
        ## import pdb; pdb.set_trace()
        name = 'RedElasMat'
        ret = name + '(['
        space = len(name)
        (rows,cols) = self.shape
        fmtstr = "%0." + str(prec) + "e"
        for i in xrange(rows):
            if i != 0: ret+= ("%" + str(space+2) + "s") % " "
            ret += "[ "
            for j in xrange(cols):
                val = self[i,j]
                if val == 0:
                    fmtlen = len(fmtstr % 0)
                    ret+= ("%" + str(fmtlen) + "s") % "0"
                else:
                    ret+= (fmtstr) % val
                if j != (cols-1): ret += ", "
            ret += "]"
            if i != (rows-1): ret += ",\n"
        ret += "])"
        return ret
    

class Anisotropic(material.Material):
    """Elastic material
    
    Attributes:
        s: full compliance matrix
        sr: reduced compliance matrix
        c: full stiffness matrix
        cr: reduced stiffness matrixrotat
    """
    
    sr = None
    
    def __get_s(self):
        return self.sr.full()
    def __set_s(self,s):
        self.sr = s.reduced()
    def __get_c(self):
        return self.cr.full()
    def __set_c(self,c):
        self.cr = c.reduced()
    def __get_cr(self):
        return np.linalg.inv(self.sr)
    def __set_cr(self,cr):
        self.sr = np.linalg.inv(cr)
    
    s = property(__get_s,__set_s)
    c = property(__get_c,__set_c)
    cr = property(__get_cr,__set_cr)
        
    def __init__(self):
        self.sr = ReducedElasticMatrix()
        pass


class Orthotropic(Anisotropic):
    #TODO tbh add constitutive constraint checking
    
    coe = None
    
    def __get_e11(self):
        return 1/self.sr[0,0]
    def __set_e11(self, value):
        self.set_orthotropic(e11=value)
    def __get_e22(self):
        return 1/self.sr[1,1]
    def __set_e22(self, value):
        self.set_orthotropic(e22=value)
    def __get_e33(self):
        return 1/self.sr[2,2]
    def __set_e33(self, value):
        self.set_orthotropic(e33=value)
    def __get_v23(self):
        return -self.sr[1,2] * self.e22
    def __set_v23(self, value):
        self.set_orthotropic(v23=value)
    def __get_v13(self):
        return -self.sr[0,2] * self.e11
    def __set_v13(self, value):
        self.set_orthotropic(v13=value)
    def __get_v12(self):
        return -self.sr[0,1] * self.e11
    def __set_v12(self, value):
        self.set_orthotropic(v12=value)
    def __get_g23(self):
        return 1/self.sr[3,3]
    def __set_g23(self, value):
        self.set_orthotropic(g23=value)
    def __get_g13(self):
        return 1/self.sr[4,4]
    def __set_g13(self, value):
        self.set_orthotropic(g13=value)
    def __get_g12(self):
        return 1/self.sr[5,5]
    def __set_g12(self, value):
        self.set_orthotropic(g12=value)
    
    e11 = property(__get_e11, __set_e11)
    e22 = property(__get_e22, __set_e22)
    e33 = property(__get_e33, __set_e33)
    v23 = property(__get_v23, __set_v23)
    v13 = property(__get_v13, __set_v13)
    v12 = property(__get_v12, __set_v12)
    g23 = property(__get_g23, __set_g23)
    g13 = property(__get_g13, __set_g13)
    g12 = property(__get_g12, __set_g12)
    
    def __init__(self, e11=None, e22=None, e33=None, v23=0, v13=0, v12=0,
                 g23=None, g13=None, g12=None):
        Anisotropic.__init__(self)
        self.sr.fill(0)
        self.coe = CoefficientOfExpansion()
        self.set_orthotropic(e11, e22, e33, v23, v13, v12, g23, g13, g12)
    
    def set_orthotropic(self, e11=None, e22=None, e33=None, v23=None,
                        v13=None, v12=None, g23=None, g13=None, g12=None):
        if self:
            #TODO tbh add checking to make sure self is Orthotropic
            if e11 is None:
                e11 = self.e11
            if e22 is None:
                e22 = self.e22
            if e33 is None:
                e33 = self.e33
            if v23 is None:
                v23 = self.v23
            if v13 is None:
                v13 = self.v13
            if v12 is None:
                v12 = self.v12
            if g23 is None:
                g23 = self.g23
            if g13 is None:
                g13 = self.g13
            if g12 is None:
                g12 = self.g12
        
        self.sr.fill(0)
        self.sr[0,0] = 1/e11
        self.sr[1,1] = 1/e22
        self.sr[2,2] = 1/e33
        self.sr[3,3] = 1/g23
        self.sr[4,4] = 1/g13
        self.sr[5,5] = 1/g12
        self.sr[0,1] = -v12/e11
        self.sr[0,2] = -v13/e11
        self.sr[1,2] = -v23/e22
        self.sr[1,0] = self.sr[0,1]
        self.sr[2,0] = self.sr[0,2]
        self.sr[2,1] = self.sr[1,2]
    

class TransverselyIsotropic(Orthotropic):
    
    def __init__(self, e11=None, v12=0, e22=None, v23=0, g12=None):
        if e22 is None:
            g23 = None
        else:
            g23 = e22/(2*(1+v23))
        Orthotropic.__init__(self, e11, e22, e22, v23, v12, v12, g23, g12, g12)
        
    def set_transversely_isotropic(self, e11=None, v12=None, e22=None, v23=None,
                                   g12=None):
        if self:
            #TODO tbh add checking to make sure self is Orthotropic
            if e11 is None:
                e11 = self.e11
            if v12 is None:
                v12 = self.v12
            if e22 is None:
                e22 = self.e22
            if v23 is None:
                v23 = self.v23
            if g12 is None:
                g12 = self.g12
        
        g23 = 2*(1+v23)/e22
        Orthotropic.set_orthotropic(e11, e22, e22, v23, v12, v12, g23, g12, g12)

class Isotropic(TransverselyIsotropic):
    """Isotropic, elastic material property definition.
    
    Longer class information for Isotropic.  Need to decide what goes here.
    
    Attributes:
        theta: material rotation (in radians).
    """
    
    def __get_e(self):
        return 1/self.sr[0,0]
    def __get_v(self):
        return -self.sr[0,1]*self.e
    def __get_g(self):
        return 1/self.sr[3,3]
    
    e = property(__get_e)
    v = property(__get_v)
    g = property(__get_g)
    
    def __init__(self, e=None, v=None, g=None):
        (e,v,g) = self._get_missing_constant(e=e, v=v, g=g)
        TransverselyIsotropic.__init__(self, e, v, e, v, g)
    
    def _get_missing_constant(self, e=None, v=None, g=None):
        if self:
            #TODO tbh add checking to make sure self is Isotropic
            if (e is None) & (v is not None) & (g is not None):
                e = g*(2*(1+v))
            elif (v is None) & (e is not None) & (g is not None):
                v = (e/(2*g))-1
            elif (g is None) & (e is not None) & (v is not None):
                g = e/(2*(1+v))
            else:
                #just pass a default value so that sr=[0]
                v=0
        return e,v,g
    
    def set_isotropic(self, e=None, v=None, g=None):
        
        #TODO tbh add constitutive constraint checking
        TransverselyIsotropic.set_transversely_isotropic(self,e, v, e, v, g)
    
    
def ConvertReducedToFull(reduced):
    """Convert reduced [6,6] matrix to full [3,3,3,3]
    
    returns: full [3,3,3,3] tensor."""
    
    full = FullElasticTensor()
    
    for m in range(6):
        for n in range(6):
            (i,j) = FillReducedIndex(m)
            (k,l) = FillReducedIndex(n)
            
            full[i,j,k,l] = reduced[m,n]
            full[j,i,k,l] = reduced[m,n]
            full[i,j,l,k] = reduced[m,n]
            full[j,i,l,k] = reduced[m,n]
    
    return full

def ConvertFullToReduced(full):
    reduced = ReducedElasticMatrix()
    
    for m in range(6):
        for n in range(6):
            (i,j) = FillReducedIndex(m)
            (k,l) = FillReducedIndex(n)
            
            reduced[m,n] = full[i,j,k,l]
    
    return reduced


def ReduceFullIndex(i,j):
    r = np.NaN
    
    if (i,j) == (0,0):
        r = 0
    elif (i,j) == (1,1):
        r = 1
    elif (i,j) == (2,2):
        r = 2
    elif ((i,j) == (1,2)) | ((i,j) == (2,1)):
        r = 3
    elif ((i,j) == (0,2)) | ((i,j) == (2,0)):
        r = 4
    elif ((i,j) == (0,1)) | ((i,j) == (1,0)):
        r = 5
    
    return r

def FillReducedIndex(i):
    r = np.NaN
    s = np.NaN
    
    if i == 0:
        (r,s) = (0,0)
    elif i == 1:
        (r,s) = (1,1)
    elif i == 2:
        (r,s) = (2,2)
    elif i == 3:
        (r,s) = (1,2)
    elif i == 4:
        (r,s) = (0,2)
    elif i == 5:
        (r,s) = (0,1)
    
    return (r,s)

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

from eng.composite import clt
import numpy as np
import scipy as sp
import sympy
from eng.material import elastic
from eng.math import space
import copy
import string

class Plate(object):
    
    _x = sympy.Symbol('x')
    _y = sympy.Symbol('y')
    _z = sympy.Symbol('z')
    
    laminate = None
    
    disp = None
    strain = None
    stress = None
    
    def __init__(self, laminate=None):
        if laminate is not None:
            self.laminate = laminate
        else:
            self.laminate = clt.Laminate()
    
    def solve(self, **kwargs):
        self._solve(**kwargs)
        self.disp = self._disp()
        self.strain = self._strain()
        self.stress = self._stress()
    a = 1
    b = None
    
    def _solve(self):
        raise NotImplementedError()
    def _disp(self):
        raise NotImplementedError()
    def _strain(self):
        raise NotImplementedError()
    def _stress(self):
        raise NotImplementedError()

class InfinitePlate(Plate):
    def __get_b(self):
        return np.inf
    b = property(__get_b)
    
    def return_at_x(self, symbolic_function, x, pts=50):
        z = self.laminate.linspace(pts)
        func = np.empty_like(z)
        for i in xrange(0,len(z)):
            func[i] = symbolic_function.subs(
                [(sympy.Symbol('x'),x),
                 (sympy.Symbol('z'),z[i])]).evalf()
        return (z, func)
    
    def return_at_z(self, symbolic_function, z, pts=100):
        x = np.linspace(-self.a/2.,self.a/2.,int(round(pts/2.)*2+1))
        func = np.empty_like(x)
        for i in xrange(0,len(x)):
            func[i] = symbolic_function.subs(
                [(sympy.Symbol('x'),x[i]),
                 (sympy.Symbol('z'),z)]).evalf()
        return (x, func)

class InfinitePlateFreeNormalCosLoad(InfinitePlate):
    q0 = 0
    
    def _disp(self):
        w = ((1/self.laminate.abd[3,3])*
             self.q0*
             (self.a/np.pi)**4*
             sympy.cos(np.pi*sympy.Symbol('x')/self.a))
        u = -sympy.Symbol('z')*sympy.diff(w,sympy.Symbol('x'))
        return sympy.Matrix([u, 0, w])
    
    def _strain(self):
        eclt = sympy.Matrix([(-sympy.Symbol('z')*
                              sympy.diff(self.disp[2],sympy.Symbol('x'),2)),
                             0,
                             0])
        return eclt
    
    def _stress(self):
        z = sympy.Symbol('z')
        x = sympy.Symbol('x')
        # constants of integration
        const_int_shear = 0
        const_int_normal = -self.q0
        top = self.laminate.convert_index_to_position(0)
        pw_args = []
        # make a piecewise function for stress
        for i in xrange(self.laminate.ply_count):
            if i == 0:
                limit = z < top
                functions = sympy.Matrix([0,0,0,0,0,0]) / 0
                pw_args.append(ripple(functions, limit))
            bot = top
            top = self.laminate.convert_index_to_position(i+1)
            sclt = self.laminate[i].qbar * self.strain
            
            sxzclt = sympy.integrate(-sympy.diff(sclt[0],x), z)
            sxzclt = sxzclt - sxzclt.subs(z,bot) + const_int_shear
            const_int_shear = sxzclt.subs(z,top)
            szzclt = sympy.integrate(-sympy.diff(sxzclt,x), z)
            szzclt = szzclt - szzclt.subs(z,bot) + const_int_normal
            const_int_normal = szzclt.subs(z,top)
            
            limits = z <= top
            functions = [sclt[0], sclt[1], szzclt, 0, sxzclt, 0]
            pw_args.append(ripple(functions, limits))
            if i == self.laminate.ply_count - 1:
                limit = z > top
                functions = sympy.Matrix([0,0,0,0,0,0]) / 0
                pw_args.append(ripple(functions, limit))
        sclt = sympy.Matrix([0,0,0,0,0,0])
        for i in xrange(sclt.rows):
            sclt[i] = sympy.Piecewise(*zip(*pw_args)[i])
        
        return sclt
    
    def _solve(self):
        pass

def ripple(list, a):
    r = []
    for i in iter(list):
        r.append((i,a))
    return r

class InfinitePlateFreeNormalCosLoad_Elasticity(InfinitePlate):
    q0 = 0
    
    def p(self):
        return np.pi / self.a
    
    def b(self, ply_number):
        ply = self.laminate[ply_number]
        b = ((ply.mat.cr[0,2]**2 + 2*ply.mat.cr[0,2]*ply.mat.cr[4,4] -
              ply.mat.cr[0,0]*ply.mat.cr[2,2])/
             (ply.mat.cr[2,2]*ply.mat.cr[4,4]))
        return b
    
    def c(self, ply_number):
        ply = self.laminate[ply_number]
        return ply.mat.cr[0,0] / ply.mat.cr[2,2]
    
    def lamb(self, i, ply_number):
        p = self.p()
        b = self.b(ply_number)
        c = self.c(ply_number)
        if i == 0:
            lamb = +np.sqrt((-b+np.sqrt(b**2-4*c))/2)*p
        elif i == 1:
            lamb = +np.sqrt((-b-np.sqrt(b**2-4*c))/2)*p
        elif i == 2:
            lamb = -np.sqrt((-b+np.sqrt(b**2-4*c))/2)*p
        elif i == 3:
            lamb = -np.sqrt((-b-np.sqrt(b**2-4*c))/2)*p
        return lamb

    def phi(self, i, ply_number):
        p = self.p()
        ply = self.laminate[ply_number]
        phi = (((ply.mat.cr[0,2] + ply.mat.cr[4,4])*p*self.lamb(i, ply_number))/
               (ply.mat.cr[4,4]*self.lamb(i, ply_number)**2 - ply.mat.cr[0,0]*p**2))
        return phi
    
    def _disp(self):
        z = sympy.Symbol('z')
        x = sympy.Symbol('x')
        top = self.laminate.convert_index_to_position(0)
        pw_args = []
        # make a piecewise function
        for i in xrange(self.laminate.ply_count):
            if i == 0:
                limit = z < top
                functions = sympy.Matrix([0,0,0]) / 0
                pw_args.append(ripple(functions, limit))
            bot = top
            top = self.laminate.convert_index_to_position(i+1)
            disp = [
                ((self.phi(0,i)*self.w[0][i]*sympy.exp(self.lamb(0,i)*z) +
                  self.phi(1,i)*self.w[1][i]*sympy.exp(self.lamb(1,i)*z) +
                  self.phi(2,i)*self.w[2][i]*sympy.exp(self.lamb(2,i)*z) +
                  self.phi(3,i)*self.w[3][i]*sympy.exp(self.lamb(3,i)*z))*
                 sympy.sin(self.p()*x)),
                0,
                ((self.w[0][i]*sympy.exp(self.lamb(0,i)*z) +
                  self.w[1][i]*sympy.exp(self.lamb(1,i)*z) +
                  self.w[2][i]*sympy.exp(self.lamb(2,i)*z) +
                  self.w[3][i]*sympy.exp(self.lamb(3,i)*z))*
                 sympy.cos(self.p()*x))]
            limits = z <= top
            pw_args.append(ripple(disp, limits))
            if i == self.laminate.ply_count - 1:
                limit = z > top
                functions = sympy.Matrix([0,0,0]) / 0
                pw_args.append(ripple(functions, limit))
        disp = sympy.Matrix([0,0,0])
        for i in xrange(disp.rows):
            disp[i] = sympy.Piecewise(*zip(*pw_args)[i])
        return disp
    
    def _strain(self):
        z = sympy.Symbol('z')
        x = sympy.Symbol('x')
        strain = sympy.Matrix([sympy.diff(self.disp[0],x),
                               0,
                               sympy.diff(self.disp[2],z),
                               0,
                               (sympy.diff(self.disp[0],z)+
                                sympy.diff(self.disp[2],x)),
                               0])
        return strain
    
    def _stress(self):
        z = sympy.Symbol('z')
        x = sympy.Symbol('x')
        props = self.laminate.piecewise_cr()
        strain = self.strain
        stress = sympy.piecewise_fold(props * self.strain)
        return stress
    
    w = None
    
    @staticmethod
    def get_w_symbol(i,j):
        return sympy.Symbol('w' + repr(i) + repr(j))
    
    def update_w(self,m,n):
        for i in xrange(self.laminate.ply_count):
            for j in xrange(4):
                self.w[j][i] = (
                    self.w[j][i].subs(
                        self.get_w_symbol(m,n),self.w[m][n]).evalf())
    
    def __init__(self, *args):
        self.reset_w()
        pass
    
    def reset_w(self):
        w = []
        for i in xrange(4):
            row = []
            for j in xrange(self.laminate.ply_count):
                row.append(self.get_w_symbol(i,j))
            w.append(row)
        self.w = w
    
    def _solve(self, status=False):
        from time import time
        self._start = time()
        self._step = self._start
        def update(val=None, i=None, j=None):
            if not status:
                if i is None:
                    print 'Started solver...'
                else:
                    self.update_w(i,j)
                    tick = time()
                    fmt = 'Solved w[%0.0f][%0.0f] at %5.2f s in %5.2f s'
                    tick_start = tick - self._start
                    tick_step = tick - self._step
                    print fmt % (i, j, tick_start, tick_step)
                    self._step = tick
            return
        ## import pdb; pdb.set_trace()
        x = sympy.Symbol('x')
        z = sympy.Symbol('z')
        self.reset_w()
        update(time())
        # first solve for bottom surface z=-H/2
        pos = self.laminate.convert_index_to_position(0)
        self.w[0][0] = sympy.solve(
            self.stress[2].subs(z,pos).evalf()+
            self.q0 * sympy.cos(self.p() * x),self.w[0][0])[0]
        update(i=0,j=0)
        self.w[1][0] = sympy.solve(
            self.stress[4].subs(z,pos).evalf(),self.w[1][0])[0]
        update(i=1,j=0)
        # then the interfaces
        for i in xrange(self.laminate.ply_count - 1):
            eps = 10**-6
            lower = self.laminate.convert_index_to_position(i+1-eps)
            upper = self.laminate.convert_index_to_position(i+1+eps)
            self.w[2][i] = sympy.solve(
                self.stress[2].subs(z,lower).evalf()-
                self.stress[2].subs(z,upper).evalf(),self.w[2][i])[0]
            update(i=2,j=i)
            self.w[3][i] = sympy.solve(
                self.stress[4].subs(z,lower).evalf()-
                self.stress[4].subs(z,upper).evalf(),self.w[3][i])[0]
            update(i=3,j=i)
            self.w[0][i+1] = sympy.solve(
                self.disp[0].subs(z,lower).evalf()-
                self.disp[0].subs(z,upper).evalf(),self.w[0][i+1])[0]
            update(i=0,j=i+1)
            self.w[1][i+1] = sympy.solve(
                self.disp[2].subs(z,lower).evalf()-
                self.disp[2].subs(z,upper).evalf(),self.w[1][i+1])[0]
            update(i=1,j=i+1)
        # finally, the upper surface z=+H/2
        ply = self.laminate.ply_count - 1
        pos = self.laminate.convert_index_to_position(ply + 1)
        self.w[2][ply] = sympy.solve(
            self.stress[2].subs(z,pos).evalf(),self.w[2][ply])[0]
        update(i=2,j=ply)
        self.w[3][ply] = sympy.solve(
            self.stress[4].subs(z,pos).evalf(),self.w[3][ply])[0]
        update(i=3,j=ply)
    
    def piecewise_cr(self):
        z = sympy.Symbol('z')
        x = sympy.Symbol('x')
        # make a piecewise function
        cr = sympy.Matrix([[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])
        for i in xrange(cr.rows):
            for j in xrange(cr.cols):
                top = self.laminate.convert_index_to_position(0)
                pw_args = []
                for k in xrange(self.laminate.ply_count):
                    if k == 0:
                        pw_args.append((sympy.nan, z < top))
                    bot = top
                    top = self.laminate.convert_index_to_position(k+1)
                    pw_args.append((self.laminate[k].mat.cr[i,j], z <= top))
                    if k == self.laminate.ply_count - 1:
                        pw_args.append((sympy.nan, z > top))
                cr[i,j] = sympy.Piecewise(*pw_args)
        return cr

class InfPlatePartialLoadLinear(InfinitePlate):
    q0 = 0
    alpha = 0
    is_shear = False
    
    def _q(self, section):
        if section is 'mid':
            res = self.q0
        else:
            res = 0
        return res
    def _div_left(self):
        return -self.alpha*self.a/2.0
    def _div_right(self):
        return self.alpha*self.a/2.0
    
    
    def _u_pw(self):
        u = sympy.Piecewise((self._u('left'), self._x <= self._div_left()),
                            (self._u('mid'), self._x < self._div_right()),
                            (self._u('right'), True))
        return u
    
    
    def _w_pw(self):
        w = sympy.Piecewise((self._w('left'), self._x <= self._div_left()),
                            (self._w('mid'), self._x < self._div_right()),
                            (self._w('right'), True))
        return w

    def _disp(self):
        return sympy.Matrix([self._u_pw(), 0, self._w_pw()])
    
    def _e(self, section):
        res = (sympy.diff(self._u(section),self._x)-
               self._z*sympy.diff(self._w(section),self._x,2))
        return res
    def _e_pw(self):
        e = sympy.Piecewise((self._e('left'), self._x <= self._div_left()),
                            (self._e('mid'), self._x < self._div_right()),
                            (self._e('right'), True))
        return e
    
    def _strain(self):
        return sympy.Matrix([self._e_pw(), 0, 0])
    def _s(self, section):
        q0 = self.q0
        z = sympy.Symbol('z')
        x = sympy.Symbol('x')
        a = self.a
        alpha = self.alpha
        # constants of integration
        if(self.is_shear):
            const_int_shear = -self._q(section)
            const_int_normal = 0
        else:
            const_int_shear = 0
            const_int_normal = -self._q(section)
        top = self.laminate.convert_index_to_position(0)
        pw_args = []
        # make a piecewise function for stress
        for i in xrange(self.laminate.ply_count):
            if i == 0:
                limit = z < top
                functions = sympy.Matrix([0,0,0,0,0,0]) / 0
                pw_args.append(ripple(functions, limit))
            bot = top
            top = self.laminate.convert_index_to_position(i+1)
            strain = sympy.Matrix([self._e(section), 0, 0])
            sclt = self.laminate[i].qbar * strain
            
            sxzclt = self.integrate_shear(sclt[0])
            sxzclt = sxzclt - sxzclt.subs(z,bot).evalf() + const_int_shear
            const_int_shear = sxzclt.subs(z,top).evalf()
            szzclt = self.integrate_normal(sclt[0], sxzclt, section)
            ## szzclt = sympy.integrate(-sympy.diff(sxzclt,x), z)
            szzclt = szzclt - szzclt.subs(z,bot).evalf() + const_int_normal
            const_int_normal = szzclt.subs(z,top).evalf()
            
            limits = z <= top
            functions = [sclt[0], sclt[1], szzclt, 0, sxzclt, 0]
            pw_args.append(ripple(functions, limits))
        
        limit = z > top
        functions = sympy.Matrix([0,0,0,0,0,0]) / 0
        pw_args.append(ripple(functions, limit))
        
        sclt = sympy.Matrix([0,0,0,0,0,0])
        for i in xrange(sclt.rows):
            sclt[i] = sympy.Piecewise(*zip(*pw_args)[i])
        
        #oops...i am supposed to integrate from z=+H/2
        sclt[2] = sclt[2] - const_int_normal
        sclt[4] = sclt[4] - const_int_shear
        
        return sclt
    def _stress(self):
        res = sympy.Matrix([0,0,0,0,0,0])
        left = self._s('left')
        right = self._s('right')
        mid = self._s('mid')
        for i in xrange(6):
            res[i] = sympy.Piecewise((left[i], self._x <= self._div_left()),
                                     (mid[i], self._x < self._div_right()),
                                     (right[i], True))
        return res
    def integrate_shear(self, sxx):
        shear = sympy.integrate(-sympy.diff(sxx,self._x), self._z)
        return shear
    def integrate_normal(self, sxx, shear, section):
        normal = sympy.integrate(-sympy.diff(shear,self._x), self._z)
        return normal

class InfPlatePartialLoadNonLinear(InfPlatePartialLoadLinear):
    # the only thing to change is the definition of strain and how
    # we integrate the interlaminar normal stress
    
    def _e(self, section):
        # this is the linear strain plus an additional term
        res = (InfPlatePartialLoadLinear._e(self, section)+
               1/2.*sympy.diff(self._w(section),self._x)**2)
        ## import pdb; pdb.set_trace()
        ## res = (sympy.diff(self._u(section),self._x)+
               ## 1/2.*sympy.diff(self._w(section),self._x)**2-
               ## self._z*sympy.diff(self._w(section),self._x,2))
        return res
    
    def integrate_normal(self, sxx, shear, section):
        omega = -sympy.diff(self._w(section),self._x)
        term1 = (-sympy.diff(omega*sxx, self._x)+
                 sympy.diff(shear,self._x))
        term2 = omega * sympy.diff(shear, self._z)
        normal = sympy.integrate(-(term1 - term2), self._z)
        ## import pdb; pdb.set_trace()
        return normal

class InfPlatePartialLoadLinear_Fixed(InfPlatePartialLoadLinear):
    def _u(self, section):
        coef = ((self.a**3*self.laminate.abd[0,3]*self.q0)/
                (24*(self.laminate.abd[0,3]**2 - self.laminate.abd[0,0]*self.laminate.abd[3,3])))
        if ((section is 'left') or (section is 'right')):
            res = (self.alpha*(self.alpha**2-6*self._x/self.a)*
                   (self._x/self.a-1/2.))
        elif section is 'mid':
            res = (self._x/self.a*
                   (self.alpha*(3-3*self.alpha+self.alpha**2)-
                    4*(self._x/self.a)**2))
        res = res * coef
        return res
    def _w(self, section):
        coef = ((self.a**4*self.q0)/
                (192*self.laminate.abd[3,3]*(self.laminate.abd[0,3]**2-
                                    self.laminate.abd[0,0]*self.laminate.abd[3,3])))
        if ((section is 'left') or (section is 'right')):
            res = (self.alpha*(1+2*self._x/self.a)*
                   ((3-self.alpha**2)*(1-2*self._x/self.a)*self.laminate.abd[0,3]**2+
                    2*(-2+self.alpha**2+4*self._x/self.a*(1+self._x/self.a))
                    *self.laminate.abd[0,0]*self.laminate.abd[3,3]))
            if section is 'right':
                res = res.subs(self._x, -self._x)
        elif section is 'mid':
            res = (self.alpha*(3-self.alpha**2)*
               (1-4*(self._x/self.a)**2)*self.laminate.abd[0,3]**2+
               (self.alpha*(2*self.alpha**2-(1/2.)*self.alpha**3-4)+
                12*(2-self.alpha)*self.alpha*(self._x/self.a)**2-
                8*(self._x/self.a)**4)*self.laminate.abd[0,0]*self.laminate.abd[3,3])
        res = res * coef
        return res
    def _solve(self):
        pass
    
class InfPlatePartialLoadLinear_Free(InfPlatePartialLoadLinear):
    def _u(self, section):
        q0 = self.q0
        a = self.a
        alpha = self.alpha
        A11 = self.laminate.abd[0,0]
        B11 = self.laminate.abd[0,3]
        D11 = self.laminate.abd[3,3]
        x = self._x
        z = self._z
        coef = (a**3*B11*q0)/(24*(B11**2 - A11*D11))
        if ((section is 'left') or (section is 'right')):
            res = -alpha*(alpha**2-12*x/a+12*(x/a)**2)/2
        elif section is 'mid':
            res = (x/a)*(3*(-2+alpha)*alpha+4*(x/a)**2)
        res = res * coef
        return res
    def _w(self, section):
        q0 = self.q0
        a = self.a
        alpha = self.alpha
        A11 = self.laminate.abd[0,0]
        B11 = self.laminate.abd[0,3]
        D11 = self.laminate.abd[3,3]
        x = self._x
        z = self._z
        coef = (a**4*q0*A11)/(96*(B11**2 - A11*D11))
        if ((section is 'left') or (section is 'right')):
            res = alpha*(1+2*(x/a))*(-2+alpha**2+4*(x/a)*(1+x/a))
            if section is 'right':
                res = res.subs(self._x, -self._x)
        elif section is 'mid':
            res = -((alpha*(8-4*alpha**2+alpha**3)+
                     24*(-2+alpha)*alpha*(x/a)**2+16*(x/a)**4)/4)
        res = res * coef
        return res
    
    def _solve(self):
        pass

class InfPlatePartialShearLoadLinear_Fixed(InfPlatePartialLoadLinear):
    
    is_shear = True
    
    def _u(self, section):
        coef = ((self.a**2*self.q0)/(8*self.laminate.abd[0,0]))
        if section is 'left':
            res = 2*self.alpha*(1+2*(self._x/self.a))
        elif section is 'right':
            res = 2*self.alpha*(1-2*(self._x/self.a))
        elif section is 'mid':
            res = -4*(self._x/self.a)**2-(self.alpha-2)*self.alpha
        res = res * coef
        return res
    def _w(self, section):
        coef = ((self.laminate.thickness*self.q0*self.a**3)/
                (96*self.laminate.abd[3,3]))
        x = (self._x/self.a)
        if section is 'left':
            res = (1+2*x)*self.alpha*(4*x+4*x**2+self.alpha**2)
        elif section is 'right':
            res = (-1+2*x)*self.alpha*(-4*x+4*x**2+self.alpha**2)
        elif section is 'mid':
            res = 2*x*(self.alpha-1)*(4*x**2+self.alpha*(self.alpha-2))
        res = res * coef
        return res
    def _solve(self):
        pass

class InfPlatePartialShearLoadNonLinear_Fixed(InfPlatePartialLoadNonLinear):
    
    is_shear = True
    
    coeff_count = None
    coeffs = None
    dcoeffs = None
    
    @property
    def disp(self):
        return self._disp()
    @property
    def stress(self):
        return self._stress()
    
    def __init__(self, lam, rr=3, coeffs=[[1,3],[3,4],[1,3]]):
        Plate.__init__(self, lam)
        self.coeff_count = list(np.array(coeffs) + rr)
        self.coeffs = self._get_blank_coeffs()
        self.dcoeffs = self._get_blank_coeffs(prefix="d")
    
    def _get_blank_coeffs(self, prefix=""):
        coeffs = []
        for i in xrange(len(self.coeff_count)):
            coeffs_i = []
            for j in xrange(len(self.coeff_count[i])):
                coeffs_j = []
                letter = string.letters.swapcase()[j]
                for k in xrange(self.coeff_count[i][j]):
                    coeffs_j.append(sympy.Symbol(prefix + letter + str(i) + str(k)))
                coeffs_i.append(coeffs_j)
            coeffs.append(coeffs_i)
        return coeffs
    
    def _get_coeff_expansion(self, i, j, x=None, d=False):
        if x is None:
            x = self._x/self.a
        if d:
            c = self.dcoeffs[i][j]
        else:
            c = self.coeffs[i][j]
        ret = 0
        for i in xrange(len(c)):
            ret += c[i] * x**i
        return ret
    
    def _u(self, section, d=False):
        x = self._x/self.a
        if section is "left" or section == 0:
            return (x+0.5)*self._get_coeff_expansion(0,0, d=d)
        elif section is "mid" or section == 1:
            return self._get_coeff_expansion(1,0, d=d)
        elif section is "right" or section == 2:
            return (x-0.5)*self._get_coeff_expansion(2,0, d=d)
    def _w(self, section, d=False):
        x = self._x/self.a
        if section is "left" or section == 0:
            return (x+0.5)*self._get_coeff_expansion(0,1, d=d)
        elif section is "mid" or section == 1:
            return self._get_coeff_expansion(1,1, d=d)
        elif section is "right" or section == 2:
            return (x-0.5)*self._get_coeff_expansion(2,1, d=d)
    def _du(self, section):
        return self._u(section, d=True)
    def _dw(self, section):
        return self._w(section, d=True)
    
    def _write_to_coeff(self, k, i, j, value, d=False):
        if d:
            self.dcoeffs[k][i][j] = value
        else:
            self.coeffs[k][i][j] = value
    
    def _solve(self):
        # i'm going to follow the mathematica document as close as possible
        
        # extra equations
        def beta(i): return -sympy.diff(self._w(i),self._x)
        def eps(i): return (sympy.diff(self._u(i),self._x) + beta(i)**2)/2
        def kappa(i): return sympy.diff(beta(i),self._x)
        def n(i): return self.laminate.abd[0,0]*eps(i)
        def m(i): return self.laminate.abd[3,3]*kappa(i)
        
        def dbeta(i): return -sympy.diff(self._dw(i),self._x)
        def deps(i): return (sympy.diff(self._du(i),self._x) + beta(i)*dbeta(i))
        def dkappa(i): return sympy.diff(dbeta(i),self._x)
        
        # continuity
        raise Exception("Not yet implemented")
        return deps

class InfPlatePartialLoadNonLinear_Free(InfPlatePartialLoadNonLinear):
    def _u(self, section):
        q0 = self.q0
        a = self.a
        alpha = self.alpha
        A11 = self.laminate.abd[0,0]
        B11 = self.laminate.abd[0,3]
        D11 = self.laminate.abd[3,3]
        x = self._x
        z = self._z
        coef = n/a
        if ((section is 'left') or (section is 'right')):
            res = n/a
        elif section is 'mid':
            res = n/a
        res = res * coef
        return res
    def _w(self, section):
        q0 = self.q0
        a = self.a
        alpha = self.alpha
        A11 = self.laminate.abd[0,0]
        B11 = self.laminate.abd[0,3]
        D11 = self.laminate.abd[3,3]
        x = self._x
        z = self._z
        coef = n/a
        if ((section is 'left') or (section is 'right')):
            res = n/a
            if section is 'right':
                res = res.subs(self._x, -self._x)
        elif section is 'mid':
            res = n/a
        res = res * coef
        return res
    
    def _solve(self):
        pass
    
class InfPlatePartialLoadNonLinear_Fixed(InfPlatePartialLoadNonLinear):
    
    beta = None
    
    @property
    def n0(self):
        D11=self.laminate.abd[3,3]
        n0 = (self.beta**2)*D11
        return n0
    
    def _solve(self, x0=8000):
        q0 = self.q0
        a = self.a
        aa = self.alpha
        A11 = self.laminate.abd[0,0]
        B11 = self.laminate.abd[0,3]
        D11 = self.laminate.abd[3,3]
        bb = sympy.Symbol('bb') #beta
        zero = ((1/(48.*A11*bb**7*D11**2))*
                (1/sympy.cosh((a*bb)/2.))**2*
                (a*bb**3*(24*bb**6*D11**3+a**2*A11*aa**2*(-3+2*aa)*q0**2)*
                 sympy.cosh((a*bb)/2.)**2+3*A11*q0**2*
                 (a*bb+4*a*aa*bb+
                  4*a*aa*bb*sympy.cosh(a*bb)+
                  a*aa*bb*sympy.cosh(a*(-1+aa)*bb)-
                  a*bb*sympy.cosh(a*aa*bb)+a*aa*bb*sympy.cosh(a*aa*bb)-
                  5*sympy.sinh(a*bb)-5*sympy.sinh(a*(-1+aa)*bb)-
                  5*sympy.sinh(a*aa*bb))))
        self.beta = float(
            sympy.mpmath.findroot(lambda x:zero.subs(bb,x).evalf(), x0))
        return
        
    def _u(self, section):
        q0 = self.q0
        a = self.a
        aa = self.alpha
        bb = self.beta
        A11 = self.laminate.abd[0,0]
        B11 = self.laminate.abd[0,3]
        D11 = self.laminate.abd[3,3]
        n0 = self.n0
        x = self._x
        
        if ((section is 'left') or (section is 'right')):
            res = ((1/(32*A11*bb**7*D11**2.))*
                   ((1/sympy.cosh((a*bb)/2.))**2*
                    (-16*bb**9*D11**3*(a-2*x)*
                     sympy.cosh((a*bb)/2.)**2+
                     A11*q0**2*(-2*a*bb+4*bb*x+
                                2*a**2*aa**2*bb**3*(a-2*x)*
                                sympy.cosh((a*bb)/2.)**2+
                                2*a*bb*sympy.cosh(a*aa*bb)-
                                4*bb*x*sympy.cosh(a*aa*bb)-
                                4*a*aa*bb*sympy.cosh((1/2.)*bb*
                                                     (a*(2+aa)-2*x))+
                                4*a*aa*bb*sympy.cosh((1/2.)*bb*
                                                     (a*(-2+aa)+2*x))-
                                4*a*aa*bb*sympy.cosh((a*aa*bb)/2.-bb*x)+
                                4*a*aa*bb*sympy.cosh((a*aa*bb)/2.+bb*x)-
                                2*sympy.sinh(bb*(a-2*x))+
                                sympy.sinh(bb*(a-a*aa-2*x))+
                                sympy.sinh(bb*(a+a*aa-2*x))))))
            if section is 'left':
                res = -res.subs(self._x, -self._x)
        elif section is 'mid':
            # this is correct
            res = ((24*bb**9*D11**3*x+
                    6*A11*bb*q0**2*x-
                    4*A11*bb**3*q0**2*x**3+
                    sympy.cosh(a*bb)*
                    (24*bb**9*D11**3*x-
                     4*A11*bb**3*q0**2*x**3-
                     3*A11*q0**2*sympy.cosh(a*aa*bb)*
                     (-2*bb*x+sympy.sinh(2*bb*x)))+
                     3*A11*q0**2*(16*sympy.cosh((a*bb)/2)**2*
                                  sympy.cosh((a*aa*bb)/2)*
                                  (bb*x*sympy.cosh(bb*x)-sympy.sinh(bb*x))-
                                  sympy.sinh(2*bb*x)+
                                  sympy.sinh(a*bb)*
                                  (8*sympy.sinh((a*aa*bb)/2)*
                                   ((-bb)*x*sympy.cosh(bb*x)+
                                    sympy.sinh(bb*x))+
                                    sympy.sinh(a*aa*bb)*
                                    (-2*bb*x+sympy.sinh(2*bb*x)))))/
                   (24*A11*bb**7*D11**2*(1+sympy.cosh(a*bb))))
        return res
    def _w(self, section):
        q0 = self.q0
        a = self.a
        aa = self.alpha
        bb = self.beta
        A11 = self.laminate.abd[0,0]
        B11 = self.laminate.abd[0,3]
        D11 = self.laminate.abd[3,3]
        n0 = self.n0
        x = self._x
        
        if ((section is 'left') or (section is 'right')):
            res = ((1/(4.*n0**2))*q0*
                   (-4*(1/sympy.cosh((1/2.)*a*bb))*
                    sympy.sinh((1/2.)*(a-2*(-x))*bb)*
                    sympy.sinh((1/2.)*a*aa*bb)*D11+a*(a-2*(-x))*aa*n0))
            if section is 'right':
                res = res.subs(self._x, -self._x)
        elif section is 'mid':
            res = ((1/(8.*n0**2))*q0*
                   (8*(-1+
                       sympy.cosh(x*bb)*sympy.cosh((1/2.)*a*(-1+aa)*bb)*
                       (1/sympy.cosh((1/2.)*a*bb)))*D11-
                       (4*x**2+a**2*(-2+aa)*aa)*n0))
        return res

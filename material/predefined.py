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

from eng.material import elastic

def carbon_fiber_generic():

    mat = elastic.TransverselyIsotropic(155e9, 0.248, 12.1e9, 0.458, 4.4e9)
    cte = elastic.CoefficientOfExpansion("thermal",2)
    cte[:] = (-0.018e-6, 24.3e-6)
    cme = elastic.CoefficientOfExpansion("moisture",2)
    cme[:] = (146.0e-6, 4770e-6)
    mat.append_coe(cte)
    mat.append_coe(cme)
    
    return mat

def glass_fiber_generic():
    mat = elastic.TransverselyIsotropic(50e9, 0.254, 15.2e9, 0.428, 4.7e9)
    cte = elastic.CoefficientOfExpansion("thermal",2)
    cte[:] = (6.34e-6, 23.3e-6)
    cme = elastic.CoefficientOfExpansion("moisture",2)
    cme[:] = (434.0e-6, 6320e-6)
    mat.append_coe(cte)
    mat.append_coe(cme)
    
    return mat

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
from eng.material import predefined
from eng.composite import layup
from eng.math import space
import scipy

def pw_cf_angleply():

    mat = predefined.carbon_fiber_generic()

    a = layup.Laminate()
    t = 150e-6
    a.append(layup.Ply(scipy.radians(30),t,mat))
    a.append(layup.Ply(scipy.radians(-30),t,mat))
    a.append(layup.Ply(scipy.radians(0),t,mat))
    a.append(layup.Ply(scipy.radians(90),t,mat))
    a.append(layup.Ply(scipy.radians(90),t,mat))
    a.append(layup.Ply(scipy.radians(0),t,mat))
    a.append(layup.Ply(scipy.radians(-30),t,mat))
    a.append(layup.Ply(scipy.radians(30),t,mat))
    
    return a

def lac_test_02():

    mat01 = predefined.carbon_fiber_generic()
    mat02 = predefined.glass_fiber_generic()

    a = layup.Laminate()
    t = 150e-6
    a.append(layup.Ply(scipy.radians(40),t,mat01))
    a.append(layup.Ply(scipy.radians(-50),t,mat02))
    a.append(layup.Ply(scipy.radians(-5),t,mat02))
    a.append(layup.Ply(scipy.radians(85),t,mat01))
    
    return a

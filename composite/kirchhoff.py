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

def from_midplane_strains(midplane_strains, z):
    """Calculate strains from midplane strains and curvatures
    
    arguments:
        midplane_strains: [6,1] ndarray, 3 strains, 3 curvatures
        z: float
    returns:
        strains: [3,1] ndarray"""
    
    size_z = numpy.size(z)
    
    
    e = midplane_strains[0:3,0].reshape((3,1)) #strains epsilon
    k = midplane_strains[3:6,0].reshape((3,1)) #curvatures
    
    strains = e + z*k
    
    pass




















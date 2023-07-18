# first, we import the necessary libraries

import numpy as np
from aart_func import * 
from params import *

"""In both Zach's and Alejandro's paper, theta refers to the inclination, and phi refers to the azimuthal again. The difference is that in Alejandro's paper, the order of spherical coordinates is given in (r, theta, phi), wheareas in Zach's paper, it is (r, phi, theta). Somewhat bizarrely, Zach's code is executed with the order (r, theta, phi). In any case, all the correction that is needed is a switch of the last two columns in the transformation matrices. In the follow code, all matrix multiplication follows the order of (t, r, theta, phi)"""

"""Zach describes (0.71, 0.71, 0) as equatorial because the hotspot is located on the 
equatorial plane, and if the magnetic field only has r and phi components, and is
on the equatorial plane, then it can be described as "equatorial" """

"""Similarly, Zach describes (0, 0, 1) as vertical for the same reason. Since the hotspot is on the equatorial plane, a magnetic field that only has theta component can be described as "vertical" """



# this function is very simple, it takes the r, phi, and theta components INDEPENDENTLY from the user, and puts them into an array. That's it!
"""notice that we are in the order of r, phi, and theta, NOT r, theta, and phi. We are using Zack's convention here."""
def b_local_array (b_r, b_phi, b_theta): 
    b_t = 0
    b_local_array = [b_t, b_r, b_phi, b_theta]
    return b_local_array

# this function returns the matrix that transforms the local magnetic field to the ZAMO magnetic field. 
# The exact formula for the matrix can be found in Appendix A equation A1 in Zack's paper (Zachary Gelles, Polarized Image of Equatorial Emission in the Kerr Geometry, 2021)
# quantities like delta, xi, and omega_zamo are found in equation 2 of Zack's paper.
def zamo_transform_matrix (r_source):
    delta = r_source**2 - 2*r_source + spin_case**2
    xi = (r_source**2 + spin_case**2)**2 - delta * (spin_case**2) # here, we are taking theta = pi/2. This is a correction from previously where I mistook theta for i_case. 
    omega_zamo = (2 * spin_case * r_source) / xi
    entry1_1 = (1/r_source) * np.sqrt(xi/delta)
    entry1_3 = (omega_zamo / r_source) * np.sqrt(xi/delta)
    entry2_2 = np.sqrt(delta) / r_source
    entry3_3 = r_source / (np.sqrt(xi))
    entry4_4 = - (1/r_source)
    zamo_transform_matrix = np.array([[entry1_1, 0, entry1_3, 0], [0, entry2_2, 0, 0], [0, 0, entry3_3, 0], [0, 0, 0, entry4_4]])
    return zamo_transform_matrix


# this function returns the matrix that transforms the ZAMO magnetic field to the observer magnetic field. 
# the exact formula to this matrix can be found in Appendix A equation A3 in Zack's paper.
# beta and kai are given in equation 46 of Zack's paper. 
def boost_transform_matrix(r_source):
    beta = (spin_case**2 - 2 * np.abs(spin_case) * np.sqrt(r_source) + r_source**2) / (np.sqrt(spin_case**2 + r_source * (r_source -2)) * (np.abs(spin_case) + r_source*1.5))
    kai = - np.pi/2
    gamma = 1/(np.sqrt(1-beta**2))
    bentry1_1 = gamma
    bentry1_2 = -beta*gamma*np.cos(kai)
    bentry1_3 = -beta*gamma*np.sin(kai)
    bentry2_1 = bentry1_2
    bentry2_2 = (gamma-1) * (np.cos(kai))**2 + 1
    bentry2_3 = (gamma-1) * np.cos(kai) * np.sin(kai)
    bentry3_1 = bentry1_3 
    bentry3_2 = bentry2_3
    bentry3_3 = (gamma-1) * (np.sin(kai))**2 + 1
    boost_transform_matrix = [ [bentry1_1, bentry1_2, bentry1_3, 0], [bentry2_1, bentry2_2, bentry2_3, 0], [bentry3_1, bentry3_2, bentry3_3, 0], [0, 0, 0, 1] ]
    return boost_transform_matrix


# this is the master function that takes in the local magnetic field and the emission radius, and returns the observer magnetic field.
# it utilizes all of the functions above. 
# I think this code is pretty easy to read, hopefully. 
def b_local_to_observer_transform(b_r, b_theta, b_phi, r_source):
    b_local_array1 = b_local_array(b_r, b_theta, b_phi)
    zamo_transform_matrix1 = zamo_transform_matrix(r_source) 
    boost_transform_matrix1 = boost_transform_matrix(r_source)
    minkowski_metric = np.diag([-1, 1, 1, 1])
    overall_transform_matrix = np.matmul(np.matmul(minkowski_metric, boost_transform_matrix1), zamo_transform_matrix1)
    b_observer_array = np.matmul(overall_transform_matrix, b_local_array1)
    b_observer_array = np.append(b_observer_array, i_case)
    return b_observer_array


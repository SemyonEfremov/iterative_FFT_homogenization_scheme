# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 15:39:06 2021

@author: Efremov-PC
"""

def PrintDeformationDataToVTK(name):
    
    vtk_file = open(name, "w")
    vtk_file.write("# vtk DataFile Version 2.0\n")
    vtk_file.write(name + "\n")
    vtk_file.write("ASCII\n")
    vtk_file.write("DATASET STRUCTURED_GRID\n")
    vtk_file.write("DIMENSIONS " +  + "\n")
    vtk_file.write("POINTS " +  + "float\n")
#%%
import sys ,os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import numpy
import meshio
def create_mesh(mesh, cell_type):
    '''Function in Meshio to read in and process the mesh '''
    cells = numpy.vstack([cell.data for cell in mesh.cells if cell.type==cell_type])
    data = numpy.hstack([mesh.cell_data_dict["gmsh:physical"][key]
                        for key in mesh.cell_data_dict["gmsh:physical"].keys() if key==cell_type])
    mesh = meshio.Mesh(points=mesh.points, cells={cell_type: cells},
                            cell_data={"name_to_read":[data]})
    return mesh

'''function call to generate 2 meshes, 3D from tetrahedrons and 2d from triangles '''
#before this will work with new files, you need to select the meshes in gmsh programm
mesh3D = meshio.read("reduced_turbine.msh")
tetra_mesh = create_mesh(mesh3D, "tetra")
meshio.write("mesh3D.xdmf", tetra_mesh)

triangle_mesh = create_mesh(mesh3D, "triangle")
meshio.write("mesh2D.xdmf", triangle_mesh)

# %%

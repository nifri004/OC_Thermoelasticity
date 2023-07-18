#%%
'''
best run on HPC with:
nohup mpirun -n 32 python3 OC_3d_turbine.py > FE.out 2> FE.err < /dev/null &
'''

import sys ,os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from asyncore import dispatcher_with_send
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

from PIL import Image
plotting = False

print("loading mesh..")
mesh = Mesh()
with XDMFFile('model/mesh3D.xdmf') as infile: #3d mesh read
    infile.read(mesh)

mesh.scale(0.00035)# (1000m radius-> 350mm radius, with approx 700mm blade radius) 

mvc = MeshValueCollection('size_t', mesh= mesh, dim = 2) #2D mesh for boundarys
with XDMFFile('model/mesh2D.xdmf') as infile:
    infile.read(mvc, 'name_to_read')
print("mesh loaded")
marker_2D = cpp.mesh.MeshFunctionSizet(mesh, mvc) #cpp = c++

E = 210e9 # e modulus
nu = 0.3 #Poissonnumber
lmbda = Constant(E*nu/((1+nu)*(1-2*nu))) #lame1
mu = Constant(E/2/(1+nu))  #lame2
rho = Constant(8050.) # density
alpha = 1.35e-5  # thermal expansion coefficient 
kappa = Constant(alpha*(2*mu + 3*lmbda))
cp = Constant(420)  # specific heat per unit volume at constant strain 
k = Constant(36) # thermal conductivity
h = Constant(20) # convective heat transfer coefficient 20 W/m^2K


Vue = VectorElement('CG', tetrahedron, 2) # displacement finite element
Vte = FiniteElement('CG', tetrahedron, 1) # temperature finite element
V = FunctionSpace(mesh, MixedElement([Vue, Vte]))

def left(x, on_boundary):
    return near(x[0], 0) and on_boundary

def bottom(x, on_boundary):
    return near(x[1], 0) and on_boundary

bc1 = DirichletBC(V.sub(0).sub(1), Constant(0.), bottom) 
#V.sub(0)= displacement domain, V.sub(0).sub(1) displacement in y-direction
bc2 = DirichletBC(V.sub(0).sub(0), Constant(0.), left) 
#V.sub(0)= displacement domain, V.sub(0).sub(0) displacement in x-direction
bcs = [bc1, bc2]

U_ = TestFunction(V)
(u_, Theta_) = split(U_)
dU = TrialFunction(V)
(du, dTheta) = split(dU)
Uold = Function(V)
(uold, Thetaold) = split(Uold)


def eps(v):
    return 1/2*(grad(v) + grad(v).T)

def sigma(v, Theta):
    return (lmbda*tr(eps(v)) - kappa*Theta)*Identity(3) + 2*mu*eps(v)

Nincr = 20
t = np.linspace(0, 1800, Nincr+1)

def time_solver(input_vec,stop_time=False,plotting=False,plot_name="1"):
    '''
    fenics solver:
    input: vector len(1,2*Nincr)

    output:max stress, min temp, u(displacement, temp), omega_vec
    '''
    temp_inp = input_vec[0:Nincr]
    omega_inp = input_vec[Nincr:Nincr+Nincr]
        
    U_ = TestFunction(V)
    (u_, Theta_) = split(U_)
    dU = TrialFunction(V)
    (du, dTheta) = split(dU)
    Uold = Function(V)
    (uold, Thetaold) = split(Uold)

    dt = Constant(0.)

    # Rotation rate and mass density
    # 60 hz 1/s
    # winkelgeschwindigkeit omega = 2PI 60

    omega = Constant(0)
    # Loading due to centripetal acceleration (rho*omega^2*x_i)
    f = Expression(("rho*omega*omega*x[0]", "rho*omega*omega*x[1]", "0.0"),
                omega=omega, rho=rho, degree=2)
     
    Temp_out = Constant(0)
    #loading due to external temperature
    temp1 = Expression(("temp"),temp=Temp_out, degree=2)
    
    ds = Measure('ds', domain=mesh, subdomain_data=marker_2D)
    
       
    #bilinear form
    a_m = inner(sigma(du, dTheta), eps(u_))*dx
    a_t = (rho*cp*(dTheta-Thetaold)/dt*Theta_ +k*dot(grad(dTheta), grad(Theta_)))*dx + h*inner(dTheta, Theta_)*ds(1)
    a = a_t + a_m
    
    #linear form
    L_m = inner(f,u_)*dx
    L_t = h*inner(temp1, Theta_)*ds(1)
    L = L_m + L_t 
    
    form = a - L
    
    U = Function(V)

    set_log_active(False)
    start_time_solver = time.time()
    max_stress = []
    min_temp = []
    max_temp = []

    Vsig = TensorFunctionSpace(mesh, "DG", degree=1) 
    #create extra FunctioSpace for Sigma; needed for  relocate stress to mesh nodes; 
    #furthermore stress depends on the gradient of the displacement field. Displacement field is of order 2.
    #stress is at best discontinous of order 1 -> DG1

    for (i, dti) in enumerate(np.diff(t)):
        if stop_time==True:
            print("FEM timestep: ",i)
        dt.assign(dti)
        Temp_out.assign(temp_inp[i]) #external temperatur input
        omega.assign(omega_inp[i])   #rotation input
        
        solve(lhs(form) == rhs(form), U, bcs) #solve FEM
        
        Uold.assign(U) 
        u, Theta = split(U)
        
        #mises calculation
        s = sigma(u,Theta) - (1./3)*tr(sigma(u,Theta))*Identity(3)
        von_Mises = sqrt(3./2*inner(s, s))
        sig2 = FunctionSpace(mesh, "DG", 1)
        von_Mises = project(von_Mises, sig2)
        max_stress.append(von_Mises.vector().norm('linf'))
            
        pro_the = project(Theta)
        min_temp.append(pro_the.vector().min()) 
        max_temp.append(pro_the.vector().max()) 
        if plotting == True:
            ############################
            # creating one file for stress, mises stress and temperature (for paraview)
            fileResults = XDMFFile("paraview_outputs/"+str(plot_name)+"MAIN"+str(i)+".xdmf")
            fileResults.parameters["flush_output"] = True
            fileResults.parameters["functions_share_mesh"] = True
            
            stress = Function(Vsig, name="Stress")
            stress.assign(project(sigma(u, Theta), Vsig))
            fileResults.write(stress, i)

            Mises_stress = Function(Vsig, name="Mises Stress")
            Mises_stress.assign(project(von_Mises, sig2))
            fileResults.write(Mises_stress, i)
            
            Temperature = Function(V, name="Temperature")
            Temperature.assign(project(Theta))
            fileResults.write(Temperature, i)
            ######################################

    max_stress = np.asarray(max_stress)
    min_temp = np.asarray(min_temp)
    max_temp = np.asarray(max_temp)
    end_time_solver = time.time()
    running_time = end_time_solver-start_time_solver
    if stop_time==True:
        print("one call in: " + str(end_time_solver-start_time_solver)+ " s")
    
    return max_stress , max_temp , U, omega_inp ,running_time

print("FEM procedure loaded")
#%%

'''Definition objective and boundaries for optimal control 
    full control of outer heat and centripetal acceleration

'''
def objective1(temp_in):
    return 1e-6*max(time_solver(temp_in)[0])

def get_acc_constraint(i):
    def fun(a):
        return -(a[Nincr+i+1]-a[Nincr+i])/t[1] +np.pi/6
    return fun

def get_heat_flux_constraint(i):
    def fun(a):
        return -(a[i+1]-a[i])/t[1] +np.pi/6
    return fun

_tmp = []
for i in range(Nincr-1):
    fun = get_acc_constraint(i)
    _tmp.append({'type': 'ineq', 'fun': fun})                     #discrete velo derivative should be smaller than some value

_tmp.append({'type': 'ineq', 'fun': lambda a: a[-Nincr-1] -750 }) #last heat control should be greater than some value
_tmp.append({'type': 'ineq', 'fun': lambda a: time_solver(a)[1][-1] -400 }) #last maximum heat in part should be greater than some value




cons = tuple(_tmp)

def callbackF(Xi):
    global Nfeval
    stress, temp ,_,_,_ = time_solver(Xi)
    print(Nfeval, "maximum heat: "+ str(temp[-1]), "maximum stress: "+ str(max(stress)))
    Nfeval += 1

    savename="iterations/iter"+str(Nfeval)
    np.save(savename, Xi)     #save array

    title = "iter: "+str(Nfeval)
    plt.figure(Nfeval)
    plt.title(title)
    plt.grid()
    plt.xlabel("time in s")
    plt.ylabel("stress in N/mm^2 / temperature in °C")
    #plt.plot(t[:-1],Xi[:Nincr],label="heat control $T_e$")
    #plt.plot(t[:-1],Xi[Nincr:],label="centripetal acceleration control $w(t)$")
    #plt.plot(t[:-1],1e-6*stress,label="maximum stress $\max_{x,y,z}\sigma_v$")
    #plt.plot(t[:-1],temp,label="maximum heat $\max_{x,y,z}T$")
    plt.plot(t,np.append(0,Xi[:Nincr]),label="heat control $T_e$")
    plt.plot(t,np.append(0,Xi[Nincr:]),label="rotational speed control $\omega(t)$")
    plt.plot(t,np.append(0,1e-6*stress),label="maximum stress $\max_{x,y,z}\sigma_v$")
    plt.plot(t,np.append(0,temp),label="maximum heat $\max_{x,y,z}T$")
    plt.legend()
    plt.savefig(savename+".png")     #save plot
    plt.close()


bnds = []
for i in range (int(Nincr-1)):
    bnds.append((0,1000))
bnds.append((0,1000))
for i in range (int(Nincr-1)):
    bnds.append((0,377))
bnds.append((376,379))
bnds = tuple(bnds)

print("optimal control bounds, cons and objective loaded")

'''optimal control of heating + rotation acceleration'''
new = False # set True if you want to make new optimization (took long time: ~1 week)
            # False for 1 FEM callwith optimal control parameters 


start_temp = np.linspace(750,750,Nincr)
start_acc = np.linspace(0,0,Nincr)
start_vec = np.array([start_temp,start_acc]).flatten()
savename="iterations/iter0"
np.save(savename, start_vec)     



if new == True:
    print("starting new Optimal control..")
    ftol = 1e-08 #1e-06
    Nfeval = 1
    for i in range(1,20): #if errors in FEM
        try:   
            print("start at iter: ",Nfeval)
            load_name = "iterations/iter"+str(Nfeval)+".npy"
            start_vec = np.load(load_name)
            sol = minimize(objective1, start_vec , args=(), method='SLSQP', jac=None,
                        hess=None, hessp=None, bounds=bnds, tol=None, constraints=cons,
                        callback=callbackF, options={"maxiter": 10000, "disp": True, 'ftol': ftol })
            np.save("3D_full_control_ftol08.npy", sol.x)
            u_opt_fullcontrol = sol.x
            iters = sol.nit
            func_evals = sol.nfev
        except:
            print('retry')
        else:
            break
else:  
    func_evals="nan"
    iters="nan"
    try:
        u_opt_fullcontrol = np.load("3D_full_control_ftol08.npy")
        print("optimal control parameter loaded")
    except:
        print("file not found: no optimal control loaded:using start vector")
        u_opt_fullcontrol = start_vec

    
#one call for plots
print("running one FEM call with optimal control parameters... ~5-30 mins")
max_stress, max_heat, U_a, omega_vec, run_time = time_solver(u_opt_fullcontrol,stop_time=True,plotting=False)
#%%

fig = plt.figure(999,figsize=(10,6))
fig.patch.set_facecolor('xkcd:white')
title = "optimal control with $\omega (t)$, $T_e(t)$"
plt.title(title)
plt.grid()
plt.xlabel("time in s")
y_label ="stress in N/mm^2 / temperature in °C / speed in rad/s"
plt.ylabel(y_label)
plt.plot(t,np.append(0,u_opt_fullcontrol[:Nincr]),label="heat control $T_e$")
plt.plot(t,np.append(0,u_opt_fullcontrol[Nincr:]),label="rotational speed control $\omega(t)$")
plt.plot(t,np.append(0,1e-6*max_stress),label="maximum stress $\max_{x,y,z}\sigma_v$")
plt.plot(t,np.append(0,max_heat),label="maximum heat $\max_{x,y,z}T$")
plt.legend()

save_name = "OC_acc_heat.png"
plt.savefig(save_name, edgecolor='white', dpi=300)

plt.close()

# %%

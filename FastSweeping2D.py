import numpy as np

class FastSweeping:
    def sweep(self,istart,iend,jstart,jend,h,mask,u):
        isize, jsize = u.shape[0], u.shape[1]

        # Traversal order
        ip = -1 if istart>iend else 1 
        jp = -1 if jstart>jend else 1

        j  = jstart-jp
        while True:
            j = j+jp
            i = istart-ip
            while True:
                i = i+ip
                uim1 = u[i+1][j] if i==0       else u[i-1][j]
                uip1 = u[i-1][j] if i==isize-1 else u[i+1][j]
                ujm1 = u[i][j+1] if j==0       else u[i][j-1]
                ujp1 = u[i][j-1] if j==jsize-1 else u[i][j+1]
                
                # Godunov Hamiltonian
                if u[i][j] > 0:
                    a = min(uim1,uip1)
                    b = min(ujm1,ujp1)

                    ab = [a,b]
                    ab.sort()

                    a, b = ab[0], ab[1]
                    unew = 0

                    utemp = a+h
                    if utemp < b:
                        unew = utemp
                    else:
                        unew = 0.5*(a+b+np.sqrt(2*h*h-(a-b)**2)) 
                    
                    if mask[i][j]==0: u[i][j]=min(unew,u[i][j])
                else:
                    a = max(uim1,uip1)
                    b = max(ujm1,ujp1)
                    
                    ab = [a,b]
                    ab.sort()

                    a,b = ab[1], ab[0]
                    unew = 0

                    utemp = a-h
                    if utemp > b:
                        unew = utemp
                    else:
                        unew = 0.5*(a+b-np.sqrt(2*h*h-(a-b)**2))

                    if mask[i][j]==0: u[i][j]=max(unew,u[i][j])
                
                if i==iend : break
            if j==jend : break
    
        return u

    def SolveEikonal(self,xmin,xmax,ymin,ymax,max_iter,mask,u):
        isize, jsize = u.shape[0], u.shape[1]

        istart, iend = 0, isize-1
        jstart, jend = 0, jsize-1

        h = (xmax-xmin)/(isize-1) # Assume h=dx=dy

        for iter in range(max_iter):
            u0 = u.copy()
            u = self.sweep(istart,iend  ,jstart,jend  ,h,mask,u)
            u = self.sweep(iend  ,istart,jstart,jend  ,h,mask,u)
            u = self.sweep(istart,iend  ,jend  ,jstart,h,mask,u)
            u = self.sweep(iend  ,istart,jend  ,jstart,h,mask,u)
            if np.linalg.norm(u0-u)*np.sqrt(h*h)<1e-8:
                print("It converges at iter=",iter)
                break

        return u
    
if __name__ == "__main__":    
    N = 17
    phi = np.zeros((N,N))
    xmin, xmax = -1,1; dx = (xmax-xmin)/(N-1)
    ymin, ymax = -1,1; dy = (ymax-ymin)/(N-1)

    isize, jsize = phi.shape[0], phi.shape[1]

    #Circle
    for i in range(isize): 
        x = xmin+i*dx
        for j in range(jsize):
            y = ymin+j*dy
            if x*x+y*y >= 0.5*0.5:
                phi[i][j] = 10000
            else:
                phi[i][j] = -10000

    # Dirichlet condition
    xexact, yexact = np.zeros(10000), np.zeros(10000)
    for i in range(10000):
       xexact[i] = 0.5*np.cos(2*np.pi*i/(10000-1))
       yexact[i] = 0.5*np.sin(2*np.pi*i/(10000-1))

    mask = np.zeros((isize,jsize))
    for i in range(isize-1):
        x = xmin+i*dx
        for j in range(jsize-1):
            y = ymin+j*dy
            phi0   = phi[i  ][j  ]
            phiip1 = phi[i+1][j  ]
            phijp1 = phi[i  ][j+1] 

            if phi0*phiip1 < 0 or phi0*phijp1 < 0:
                mask[i][j] = 1
                
                distmin = 10000
                for k in range(10000):
                    dist = np.sqrt((x-xexact[k])*(x-xexact[k])+(y-yexact[k])*(y-yexact[k]))
                    if distmin >= dist: distmin=dist
                
                if phi0<0 : phi[i][j]=-distmin
                else      : phi[i][j]= distmin

    solver = FastSweeping()
    u =  solver.SolveEikonal(xmin,xmax,ymin,ymax,100,mask,phi)
    
    error = np.zeros((N,N))
    for i in range(N):
        x = xmin+i*dx
        for j in range(N):
            y = ymin+j*dy
            error[i][j] = np.abs(u[i][j] - (np.sqrt(x*x+y*y)-0.5))

    print("L2 error = ", np.linalg.norm(error)*np.sqrt(dx*dx))

    import matplotlib.pyplot as plt    

    grid_x = np.linspace(xmin,xmax,N)
    grid_y = np.linspace(ymin,ymax,N)
    X, Y = np.meshgrid(grid_x, grid_y)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plotting the basic 3D surface
    #ax.plot_surface(X[9:,8:],Y[8:,8:],error[8:,8:],cmap='viridis')
    ax.plot_surface(X,Y,u,cmap='viridis')
    plt.show()


        

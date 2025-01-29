import numpy as np

class FastSweeping3D:
    def sweep(self,istart,iend,jstart,jend,kstart,kend,h,mask,u):
        isize, jsize, ksize = u.shape[0], u.shape[1], u.shape[2]

        # Traversal order
        ip = -1 if istart>iend else 1 
        jp = -1 if jstart>jend else 1
        kp = -1 if kstart>kend else 1

        k  = kstart-kp
        while True:
            k  = k+kp
            j  = jstart-jp
            while True:
                j = j+jp
                i = istart-ip
                while True:
                    i = i+ip
                    uim1 = u[i+1][j][k] if i==0       else u[i-1][j][k]
                    uip1 = u[i-1][j][k] if i==isize-1 else u[i+1][j][k]
                    ujm1 = u[i][j+1][k] if j==0       else u[i][j-1][k]
                    ujp1 = u[i][j-1][k] if j==jsize-1 else u[i][j+1][k]
                    ukm1 = u[i][j][k+1] if k==0       else u[i][j][k-1]
                    ukp1 = u[i][j][k-1] if k==ksize-1 else u[i][j][k+1]

                    # Godunov Hamiltonian
                    if u[i][j][k] > 0:
                        a = min(uim1,uip1)
                        b = min(ujm1,ujp1)
                        c = min(ukm1,ukp1)

                        test = [a,b,c]
                        test.sort()

                        a,b,c = test[0], test[1], test[2]

                        unew = 0

                        utemp = a+h
                        if utemp < b:
                            unew = utemp 
                        else:
                            utemp = 0.5*(a+b+np.sqrt(2*h*h-(a-b)**2))
                            if utemp < c:
                                unew = utemp
                            else:
                                unew = (a+b+c+np.sqrt((a+b+c)**2-3*(a*a+b*b+c*c-h*h)))/3.0
                        
                        if mask[i][j][k] == 0 : u[i][j][k]=min(unew,u[i][j][k])
                    else:
                        a = max(uim1,uip1)
                        b = max(ujm1,ujp1)
                        c = max(ukm1,ukp1)

                        test = [a,b,c]
                        test.sort()

                        a,b,c = test[2], test[1], test[0]

                        unew = 0

                        utemp = a-h
                        if utemp > b:
                            unew = utemp 
                        else:
                            utemp = 0.5*(a+b-np.sqrt(2*h*h-(a-b)**2))
                            if utemp > c:
                                unew = utemp
                            else:
                                unew = (a+b+c-np.sqrt((a+b+c)**2-3*(a*a+b*b+c*c-h*h)))/3.0
                        
                        if mask[i][j][k] == 0: u[i][j][k]=max(unew,u[i][j][k])                    
                    if i==iend : break
                if j==jend : break
            if k==kend : break

        return u

    def SolveEikonal(self,xmin,xmax,ymin,ymax,zmin,zmax,max_iter,mask,u):
        isize, jsize,ksize = u.shape[0], u.shape[1], u.shape[2]

        istart, iend = 0, isize-1
        jstart, jend = 0, jsize-1
        kstart, kend = 0, ksize-1

        h = (xmax-xmin)/(isize-1) # Assume h=dx=dy

        for iter in range(max_iter):
            u0 = u.copy()
            u = self.sweep(istart,iend  ,jstart,jend  ,kstart,kend  ,h,mask,u)
            u = self.sweep(istart,iend  ,jstart,jend  ,kend  ,kstart,h,mask,u)
            u = self.sweep(istart,iend  ,jend  ,jstart,kstart,kend  ,h,mask,u)
            u = self.sweep(istart,iend  ,jend  ,jstart,kend  ,kstart,h,mask,u)
            u = self.sweep(iend  ,istart,jstart,jend  ,kstart,kend  ,h,mask,u)
            u = self.sweep(iend  ,istart,jend  ,jstart,kstart,kend  ,h,mask,u)
            u = self.sweep(iend  ,istart,jstart,jend  ,kend  ,kstart,h,mask,u)
            u = self.sweep(iend  ,istart,jend  ,jstart,kend  ,kstart,h,mask,u)
            
            if np.linalg.norm(u0-u)*np.sqrt(h*h*h)<1e-10:
                print("It converges at iter=",iter)
                break

        return u
    
if __name__ == "__main__":    
    N = 65
    phi = np.zeros((N,N,N))
    xmin, xmax = -1,1; dx = (xmax-xmin)/(N-1)
    ymin, ymax = -1,1; dy = (ymax-ymin)/(N-1)
    zmin, zmax = -1,1; dz = (zmax-zmin)/(N-1)

    isize, jsize, ksize = phi.shape[0], phi.shape[1], phi.shape[2]

    #Sphere
    for i in range(isize): 
        x = xmin+i*dx
        for j in range(jsize):
            y = ymin+j*dy
            for k in range(ksize):
                z = zmin+k*dz
                if x*x+y*y+z*z >= 0.5*0.5:
                    phi[i][j][k] = 10000
                else:
                    phi[i][j][k] = -10000

    # Dirichlet condition
    ind = 0
    DN = 100
    xexact, yexact, zexact = np.zeros(DN*DN), np.zeros(DN*DN), np.zeros(DN*DN)
    
    for i in range(DN):
       for j in range(DN):
        xexact[ind] = 0.5*np.cos(np.pi*i/(DN-1))*np.cos(2*np.pi*j/(DN-1))
        yexact[ind] = 0.5*np.sin(np.pi*i/(DN-1))*np.cos(2*np.pi*j/(DN-1))
        zexact[ind] = 0.5*np.sin(2*np.pi*j/(DN-1))
        ind = ind+1

    mask = np.zeros((isize,jsize,ksize))
    count = 0
    for i in range(isize-1):
        x = xmin+i*dx
        for j in range(jsize-1):
            y = ymin+j*dy
            for k in range(ksize-1):
                z = zmin+k*dz
                phi0   = phi[i  ][j  ][k  ]
                phiip1 = phi[i+1][j  ][k  ]
                phijp1 = phi[i  ][j+1][k  ]
                phikp1 = phi[i  ][j  ][k+1] 
                
                if phi0*phiip1 < 0 or phi0*phijp1 < 0 or phi0*phikp1 < 0:
                    count = count + 1
                    mask[i][j][k] = 1
                    
                    distmin = 10000
                    for l in range(DN*DN):
                        dist = np.sqrt((x-xexact[l])*(x-xexact[l])+(y-yexact[l])*(y-yexact[l])+(z-zexact[l])*(z-zexact[l]))
                        if distmin >= dist: distmin=dist
                    
                    if phi0<0 : phi[i][j][k]=-distmin
                    else      : phi[i][j][k]= distmin

    
    solver = FastSweeping3D()
    u =  solver.SolveEikonal(xmin,xmax,ymin,ymax,zmin,zmax,100,mask,phi)
    
    error = np.zeros((N,N,N))
    for i in range(N):
        x = xmin+i*dx
        for j in range(N):
            y = ymin+j*dy
            for k in range(N):
                z = zmin+k*dz
                error[i][j][k] = np.abs(u[i][j][k] - (np.sqrt(x*x+y*y+z*z)-0.5))
                
    print(f"L2 error = {np.linalg.norm(error)*np.sqrt(dx*dx*dx):.3f}")

    
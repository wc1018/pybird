#from pybird.module import *
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
# from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
# import jax
# from jax.numpy import exp,log,linspace
# jax.config.update("jax_enable_x64", True)
from scipy.integrate import odeint
from scipy.special import erf
from numpy import exp,log,linspace,array,sqrt,pi

# numerical D,DD+,D-,DD-
class GreenFunction(object):

    def __init__(self, Omega0_m, fluid_equation_of_state = 'w0wa',EoS_dict=None, quintessence=False, EFTDE=False,parameterizations='propto_omega',
                 Omega0_k=0., vectorize=False,
                 xin=-12.,xfin=0.):
        self.vectorize = vectorize
        self.Omega0_m = Omega0_m
        if self.vectorize:
            self.Omega0_m = np.array(Omega0_m)
            self.Omega0_k = np.array(Omega0_k)
            self.w = np.array(w) if w is not None else None
        self.OmegaL_by_Omega_m = (1.-self.Omega0_m-Omega0_k)/self.Omega0_m
        
        self.xfin = xfin # the finial time used to set initial condition for decay mode
        self.x0 = xin # the initial time for ODE
        self.lo = exp(self.x0 )  # the lowest integration bound of scale factor x, x=lna

        
        self.quintessence = quintessence
        self.EFTDE = EFTDE


        self.fluid_equation_of_state = fluid_equation_of_state
        if self.fluid_equation_of_state == 'w0wa':
            self.w0 = EoS_dict['w0']
            self.wa = EoS_dict['wa']
        elif self.fluid_equation_of_state == 'chebyshev':
            self.zmax = 3.5 # for chebyshev
            self.zmin=0
            self.T0 = lambda x: 1
            self.T1 = lambda x: x
            self.T2 = lambda x: 2*x**2-1
            self.T3 = lambda x: 4*x**3-3*x
            self.c0 = EoS_dict['c0']
            self.c1 = EoS_dict['c1']
            self.c2 = EoS_dict['c2']
            self.c3 = EoS_dict['c3']
            self.B0 = 1-self.c0-self.c1-self.c2-self.c3
            self.B1 = -2*(self.c1+4*self.c2+9*self.c3)*(1+self.zmax)/self.zmax
            self.delta = 1

        self.epsrel = 1e-4
        self.H0 =68 # this is irrelevant about the result; note though there are terms in EFTDE like dalphaB/dt / H, this is equal to dalphaB/dx, thus H0 still irrelevant


        if self.EFTDE: self.EFTDE_alphas(EoS_dict=EoS_dict,parameterizations=parameterizations)
        self.interpD()


    def EFTDE_alphas(self,EoS_dict=None,parameterizations='propto_omega'):
        EFTDE_params=['alphaB','alphaT','alphaM','alphaV1','alphaV2','alphaV3']
        EFTDE_value = [EoS_dict[k] if k in EoS_dict.keys() else 0 for k in EFTDE_params]
        if set(EFTDE_value) == {0}:
            # if EFTDE_value are all zero, the ode will break, we then enforece alphaB=1
            EFTDE_value[0]=1.
            print('You input no EFT parameters, we use default 1,0,0,0,0,0')
        
        if parameterizations=='constant':
            self.alpha_B = lambda x : EFTDE_value[0]
            self.alpha_T = lambda x : EFTDE_value[1]
            self.alpha_M = lambda x : EFTDE_value[2]
            self.alpha_V1 = lambda x : EFTDE_value[3]
            self.alpha_V2 = lambda x : EFTDE_value[4]
            self.alpha_V3 = lambda x : EFTDE_value[5]

            self.dalpha_Bdx = lambda x : 0
            self.dalpha_Tdx = lambda x : 0
            self.dalpha_Mdx = lambda x : 0
            self.dalpha_V1dx = lambda x : 0
            self.dalpha_V2dx = lambda x : 0
            self.dalpha_V3dx = lambda x : 0
        elif parameterizations == 'propto_omega':
            # alpha = gamma*Omega_{dark energy}
            self.alpha_B = lambda x : EFTDE_value[0]*self.Ode(x)
            self.alpha_T = lambda x : EFTDE_value[1]*self.Ode(x)
            self.alpha_M = lambda x : EFTDE_value[2]*self.Ode(x)
            self.alpha_V1 = lambda x : EFTDE_value[3]*self.Ode(x)
            self.alpha_V2 = lambda x : EFTDE_value[4]*self.Ode(x)
            self.alpha_V3 = lambda x : EFTDE_value[5]*self.Ode(x)

            self.dalpha_Bdx = lambda x : EFTDE_value[0]*self.dOdedx(x)
            self.dalpha_Tdx = lambda x : EFTDE_value[1]*self.dOdedx(x)
            self.dalpha_Mdx = lambda x : EFTDE_value[2]*self.dOdedx(x)
            self.dalpha_V1dx = lambda x : EFTDE_value[3]*self.dOdedx(x)
            self.dalpha_V2dx = lambda x : EFTDE_value[4]*self.dOdedx(x)
            self.dalpha_V3dx = lambda x : EFTDE_value[5]*self.dOdedx(x)
        

    def ksai(self,x): return self.alpha_B(x)*(1+self.alpha_T(x))+self.alpha_T(x)-self.alpha_M(x)
    def nu(self,x): 
        dHdt_over_H2 = 3/2*(-self.w(x)*(1-self.Om(x))-1)
        return -(
                (1+self.alpha_B(x))*(
                    self.alpha_B(x)*(1+self.alpha_T(x))+self.alpha_T(x)-self.alpha_M(x)+dHdt_over_H2)
                    +self.dalpha_Bdx(x)+3/2*self.Om(x)
                    )
    def C2(self,x): return -self.nu(x)-self.alpha_B(x)*(self.ksai(x)+self.alpha_T(x)-self.alpha_M(x))
    def C3(self,x): 
        dHdt_over_H2 = 3/2*(-self.w(x)*(1-self.Om(x))-1)
        return -self.alpha_T(x)-self.alpha_V2(x)*(1-self.alpha_M(x))-self.alpha_V2(x)*dHdt_over_H2+self.dalpha_V2dx(x)
    def C4(self,x): 
        dHdt_over_H2 = 3/2*(-self.w(x)*(1-self.Om(x))-1)
        return -4*self.alpha_B(x)+2*self.alpha_M(x)-3*self.alpha_T(x)-(self.alpha_V1(x)+self.alpha_V2(x))*(1-self.alpha_M(x))-3*self.alpha_V2(x)*dHdt_over_H2+self.dalpha_V1dx(x)+self.dalpha_V2dx(x)
    def C5(self,x):
        dHdt_over_H2 = 3/2*(-self.w(x)*(1-self.Om(x))-1)
        return 3*(self.alpha_T(x)-self.alpha_V1(x)+self.alpha_V2(x)+self.alpha_V3(x))-(3*self.alpha_V2(x)+self.alpha_V3(x))*self.alpha_M(x)+(3*self.alpha_V2(x)+self.alpha_V3(x))*dHdt_over_H2-3*self.dalpha_V2dx(x)-self.dalpha_V3dx(x)
    
    def mu(self, x):
        if self.EFTDE: 
            # alphaW=1.5*(1+self.w(x))*(1-self.Om(x))
            # nu = -self.alphaB**2+self.alphaB*(alphaW-1-1.5*self.Om(x))+alphaW-self.p*self.alphaB
            # return 1.+ self.alphaB**2/nu
            return 1+self.alpha_T(x)+self.ksai(x)**2/self.nu(x)
        else: return 1.
        #return 1.
        
    def mu_Psi(self,x): return 1+self.ksai(x)*self.alpha_B(x)/self.nu(x)
    def mu_chi(self,x): return self.ksai(x)/self.nu(x)
    def mu2(self, x): return self.mu_chi(x)/4*(6*self.mu(x)*self.mu_Psi(x)*self.alpha_V2(x)+3*self.mu_chi(x)*self.mu(x)*self.alpha_V1(x)+3*self.mu_chi(x)*self.mu_Psi(x)*self.C3(x)+self.mu_chi(x)**2*self.C4(x))
    def mu22(self, x): return 1/8*(5*self.mu(x)*self.mu_chi(x)**2*(self.mu_chi(x)*self.alpha_V1(x)+2*self.mu_Psi(x)*self.alpha_V2(x))**2+
                                   2*self.mu_chi(x)**3*(3*self.mu_Psi(x)*self.C3(x)+self.mu_chi(x)*self.C4(x))*(self.mu_chi(x)*self.alpha_V1(x)+2*self.mu_Psi(x)*self.alpha_V2(x))+
                                   1/self.nu(x)*(2*self.alpha_V2(x)*self.mu(x)*(2*self.mu_Psi(x)-1)+2*self.alpha_V1(x)*self.mu_chi(x)*self.mu(x)+(3*self.mu_Psi(x)-1)*self.C3(x)*self.mu_chi(x)+self.C4(x)*self.mu_chi(x)**2)**2)
    def mu3(self,x): return 0
                

    
    

    def w(self,x):
        a = exp(x)
        if self.fluid_equation_of_state == 'w0wa':
            return self.w0+self.wa*(1-a)
        elif self.fluid_equation_of_state == 'chebyshev':
            z = 1/a-1
            x = 1-2*(self.zmax-z)/(self.zmax-self.zmin)
            u = log((1+z)/(1+self.zmax))
            if z>self.zmax:
                return -1+(self.B0+self.B1*u)*exp(-u**2/self.delta**2)
            else:
                return -1*(self.c0*self.T0(x)+self.c1*self.T1(x)+self.c2*self.T2(x)+self.c3*self.T3(x))

    
    def dwdx(self,x):
        a = exp(x)
        if self.fluid_equation_of_state == 'w0wa':
            return -self.wa*a
        elif self.fluid_equation_of_state == 'chebyshev':
            z = 1/a-1
            #x = 1-2*(self.zmax-z)/(self.zmax-self.zmin)
            u = log((1+z)/(1+self.zmax))
            if z>self.zmax:
                dwda = exp(-u**2/self.delta**2)*(-self.B1*self.delta**2+2*self.B0*u+2*self.B1*u**2)/(a**2*self.delta**2*(1+z))
                return dwda*a
            else:
                dwda = (96*self.c3*z**2+16*(self.c2-6*self.c3)*z*self.zmax+2*(self.c1-4*self.c2+9*self.c3)*self.zmax**2)/(a**2*self.zmax**2)
                return dwda*a
    
    def hw(self,x):
        # 1/lna int_a^{a_0}dlna w= 1/lna* int_0^z w/(1+z)dz
        a = exp(x)
        if self.fluid_equation_of_state == 'w0wa':
            return (self.wa-a*self.wa+(self.w0+self.wa)*x)/(x+1e-20)
        elif self.fluid_equation_of_state == 'chebyshev':
            z = 1/a-1
            u = log((1+z)/(1+self.zmax))
            def integralhw(zx):
                ax = 1/(1+zx)
                return (1 / (3 * self.zmax**3 * log(ax))) * (2 * zx * (
            3 * self.zmax * (-self.c1 * self.zmax + self.c2 * (4 + 4 * self.zmax - 2 * zx)) +
            self.c3 * (-3 * (4 + 3 * self.zmax)**2 + 12 * (2 + 3 * self.zmax) * zx - 16 * zx**2)) +
            3 * ( -8 * self.c2 * self.zmax + self.zmax**2 * (-8 * self.c2 - (self.c0 + self.c2) * self.zmax + self.c1 * (2 + self.zmax)) +
            self.c3 * (2 + self.zmax) * (16 + self.zmax * (16 + self.zmax))) * log(1 + zx))
            if z>self.zmax:
                return 0.5*self.B1*self.delta**2*(1-exp(-u**2/self.delta**2))+0.5*self.B0*self.delta*sqrt(pi)*erf(log(u/self.delta))+log((1+self.zmax)/(1+z))+integralhw(self.zmax)
            else:
                return integralhw(z)
    
    def H(self,x):
        a = exp(x)
        return self.H0*(self.Omega0_m*a**(-3)+(1.-self.Omega0_m)*a**(-3-3*self.hw(x)))**0.5
    
    def dHdx(self,x):
        return -1.5*self.H(x)*self.Om(x)-1.5*(1+self.w(x))*self.H(x)*self.Ode(x)
    
    def d2Hdx2(self,x):
        return -1.5*self.dHdx(x)*self.Om(x)-1.5*self.H(x)*self.dOmdx(x)-1.5*self.dwdx(x)*self.H(x)*self.Ode(x)-1.5*(1+self.w(x))*self.dHdx(x)*self.Ode(x)-1.5*(1+self.w(x))*self.H(x)*self.dOdedx(x)
        # return -1.5*(self.dOmdx(x)*self.C(x)*self.H(x)
        #              +self.Om(x)*self.dCdx(x)*self.H(x)
        #              +self.Om(x)*self.C(x)*self.dHdx(x))
    def Om(self,x):
        a = exp(x)
        return self.Omega0_m*a**-3/(self.H(x)**2/self.H0**2)
    
    def Ode(self,x):
        a = exp(x)
        return (1.-self.Omega0_m)*a**(-3.-3.*self.hw(x))/(self.H(x)**2/self.H0**2)
    
    def dOmdx(self,x):
        #return 3*self.Om(x)*self.Ode(x)*(1+self.w(x))+3*self.Om(x)*(self.Om(x)-1)
        return 3*self.w(x)*self.Om(x)*self.Ode(x)
    
    def dOdedx(self,x):
        #return 3*self.Ode(x)*self.Om(x)+3*self.Ode(x)*(1+self.w(x))*(self.Ode(x)-1)
        return -3*self.w(x)*self.Om(x)*self.Ode(x)
    
    def C(self, x):
        if self.quintessence: return 1. + (1.+self.w(x)) * self.Ode(x)/self.Om(x)
        else: return 1.

    def dCdx(self,x):
        if self.quintessence: return self.Ode(x)/self.Om(x)*(-3*self.w(x)*(1+self.w(x))+self.dwdx(x))
        else: return 0.


            


    
    def get_ini(self,xi,xfin=None):
        ai = exp(xi)
        afin  = exp(xfin)
        if self.EFTDE:
            if xfin>0:
                Di = ai
                dDi = ai
                Dminusi = afin**(-2)
                dDminusi = -2.*afin**(-2.)
            else:
                Di = ai
                dDi = ai
                Dminusi = ai**(-3/2)
                dDminusi = -3./2.*ai**(-3./2.)

        else:
            Di = ai
            dDi = ai
            Dminusi = ai**(-3/2)
            dDminusi = -3./2.*ai**(-3./2.)
        return [dDi,Di],[dDminusi,Dminusi]  
    

    def vector_field_p(self,y,x):
        a = exp(x)
        dD,D= y

        if self.EFTDE:
            epsilon = - self.dHdx(x)/self.H(x)
            F = 1.5*self.Om(x) *self.mu(x)
        else:
            epsilon = - self.dHdx(x)/self.H(x)+self.dCdx(x)/self.C(x)
            #epsilon = 2-0.5*(1-3*self.w(x)*self.Ode(x))+self.dCdx(x)/self.C(x)
            F = 1.5*self.Om(x) *self.C(x)#self.dHdx(x)**2/self.H(x)**2+self.d2Hdx2(x)/self.H(x)+2*self.dHdx(x)/self.H(x)-self.dCdx(x)/self.C(x)*self.dHdx(x)/self.H(x)
            #F = -self.dHdx(x)/self.H(x) 
            # this eqaution only valid for wcdm-cq not wcdm or w0wa
            #F = self.dHdx(x)**2/self.H(x)**2+self.d2Hdx2(x)/self.H(x)+2*self.dHdx(x)/self.H(x)-self.dCdx(x)/self.C(x)*self.dHdx(x)/self.H(x)
        D_D = dD
        D_dD = (epsilon-2.)*dD+F*D
        D_y = [D_dD,D_D]
        return D_y  
    
    def vector_field_m(self,y,x):
        a = exp(x)
        dDminus,Dminus= y
        if self.EFTDE:
            epsilon = - self.dHdx(x)/self.H(x)
            F = 1.5*self.Om(x) *self.mu(x)
        else:
            epsilon = - self.dHdx(x)/self.H(x)+self.dCdx(x)/self.C(x)
            #epsilon = 2-0.5*(1-3*self.w(x)*self.Ode(x))+self.dCdx(x)/self.C(x)
            F = 1.5*self.Om(x) *self.C(x)#self.dHdx(x)**2/self.H(x)**2+self.d2Hdx2(x)/self.H(x)+2*self.dHdx(x)/self.H(x)-self.dCdx(x)/self.C(x)*self.dHdx(x)/self.H(x)
            #F = -self.dHdx(x)/self.H(x) 
            # this eqaution only valid for wcdm-cq not wcdm or w0wa
            #F = self.dHdx(x)**2/self.H(x)**2+self.d2Hdx2(x)/self.H(x)+2*self.dHdx(x)/self.H(x)-self.dCdx(x)/self.C(x)*self.dHdx(x)/self.H(x)

        D_Dminus = dDminus
        D_dDminus = (epsilon-2.)*dDminus+F*Dminus
        D_y = [D_dDminus,D_Dminus]
        return D_y  
    
    def interpD(self):
        x0 = self.x0
        x1 = self.xfin
        x = linspace(x0, x1, 500)

        if self.EFTDE:
            if self.xfin>0:
                y0p,y0m = self.get_ini(x0,x1)
                sol_p = odeint(self.vector_field_p, array(y0p), x).T
                sol_m = odeint(self.vector_field_m, array(y0m), x[::-1]).T

                self.Darr = sol_p[1]
                self.dDarr = sol_p[0]/exp(x)  #dDda
                self.Dminusarr1 = sol_m[1][::-1]
                self.dDminusarr = sol_m[0][::-1]
                self.Dminusarr = self.Dminusarr1/(self.Dminusarr1[0]/(exp(-3*self.x0/2)))
                self.dDminusarr = self.dDminusarr/(self.Dminusarr1[0]/(exp(-3*self.x0/2)))/exp(x)
            else:
                y0p,y0m = self.get_ini(x0,x1)
                sol_p = odeint(self.vector_field_p, array(y0p), x).T
                sol_m = odeint(self.vector_field_m, array(y0m), x).T

                self.Darr = sol_p[1]
                self.dDarr = sol_p[0]/exp(x)  #dDda
                self.Dminusarr = sol_m[1]
                self.dDminusarr = sol_m[0]/exp(x)

        else:
            y0p,y0m = self.get_ini(x0,x1)
            sol_p = odeint(self.vector_field_p, array(y0p), x).T
            sol_m = odeint(self.vector_field_m, array(y0m), x).T

            self.Darr = sol_p[1]
            self.dDarr = sol_p[0]/exp(x)  #dDda
            self.Dminusarr = sol_m[1]
            self.dDminusarr = sol_m[0]/exp(x)

        self.D = CubicSpline(x,self.Darr)
        self.DD = CubicSpline(x,self.dDarr)  #dDda(x)
        self.Dminus = CubicSpline(x,self.Dminusarr)
        self.DDminus = CubicSpline(x,self.dDminusarr)

    # def vector_field(self,x, y,args):
    #     a = exp(x)
    #     dD,D, dDminus,Dminus= y
        
    #     epsilon = - self.dHdx(x)/self.H(x)+self.dCdx(x)/self.C(x)
    #     #epsilon = 2-0.5*(1-3*self.w(x)*self.Ode(x))+self.dCdx(x)/self.C(x)
        
    #     F = 1.5*self.Om(x) *self.C(x)#self.dHdx(x)**2/self.H(x)**2+self.d2Hdx2(x)/self.H(x)+2*self.dHdx(x)/self.H(x)-self.dCdx(x)/self.C(x)*self.dHdx(x)/self.H(x)
        
    #     #F = -self.dHdx(x)/self.H(x) 
    #     # this eqaution only valid for wcdm-cq not wcdm or w0wa
    #     #F = self.dHdx(x)**2/self.H(x)**2+self.d2Hdx2(x)/self.H(x)+2*self.dHdx(x)/self.H(x)-self.dCdx(x)/self.C(x)*self.dHdx(x)/self.H(x)


    #     D_D = dD
    #     D_dD = (epsilon-2)*dD+F*D

    #     D_Dminus = dDminus
    #     D_dDminus = (epsilon-2)*dDminus+F*Dminus

        
    #     D_y = D_dD,D_D,D_dDminus,D_Dminus
    #     return D_y  
    
    # def interpD(self):
        
    #     stepsize_controller = PIDController(rtol=1e-8, atol=1e-10)#ConstantStepSize()#
    #     term = ODETerm(self.vector_field)
    #     solver = Dopri5()
    #     x0 = self.x0
    #     x1 = 0.
    #     dt0 = None
    #     y0 = self.get_ini(x0)
    #     saveat = SaveAt(ts=linspace(x0, x1, 500))
    #     sol = diffeqsolve(term, solver, x0, x1, dt0, y0, saveat=saveat,stepsize_controller=stepsize_controller)
        
    #     x = sol.ts

    #     self.Darr = sol.ys[1]
    #     self.dDarr = sol.ys[0]/exp(x)  #dDda
    #     self.Dminusarr = sol.ys[3]
    #     self.dDminusarr = sol.ys[2]/exp(x)
        
    #     self.D = CubicSpline(x,self.Darr)
    #     self.DD = CubicSpline(x,self.dDarr)  #dDda(x)
    #     self.Dminus = CubicSpline(x,self.Dminusarr)
    #     self.DDminus = CubicSpline(x,self.dDminusarr)
        

    def fplus(self, x):
        """Growth rate"""
        return exp(x)*self.DD(x) / self.D(x)

    def fminus(self, x):
        """Decay rate"""
        return exp(x)*self.DDminus(x) / self.Dminus(x)

    def W(self,x):
        """Wronskian"""
        return (self.DDminus(x) * self.D(x) - self.DD(x) * self.Dminus(x))

 #greens functions
    def G1d(self, a, ai):
        x = log(a)
        xi = log(ai)
        return(self.DDminus(xi)*self.D(x)-self.DD(xi)*self.Dminus(x))/(ai*self.W(xi))
    def G2d(self, a, ai):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*(self.Dminus(x)*self.D(xi)-self.D(x)*self.Dminus(xi))/(ai*ai*self.W(xi))
    def G1t(self, a, ai):
        x = log(a)
        xi = log(ai)
        return a*(self.DDminus(xi)*self.DD(x)-self.DD(xi)*self.DDminus(x))/(self.fplus(x)*ai*self.W(xi))
    def G2t(self, a, ai):
        x = log(a)
        xi = log(ai)
        return a*self.fplus(xi)*(self.DDminus(x)*self.D(xi)-self.DD(x)*self.Dminus(xi))/(self.fplus(x)*ai*ai*self.W(xi))

    # second order coefficients
    def I1d(self, ai, a):
        x = log(a)
        xi = log(ai)
        if self.EFTDE: return (self.G1d(a,ai)*self.fplus(xi) + self.G2d(a,ai)*self.mu2(xi)*(1.5*self.Om(xi))**2/self.fplus(xi))*self.D(xi)**2/self.D(x)**2 / self.C(xi)
        else: return self.fplus(xi)*self.D(xi)**2*self.G1d(a,ai)/self.D(x)**2 / self.C(xi)
    def I2d(self, ai, a):
        x = log(a)
        xi = log(ai)
        if self.EFTDE: return self.G2d(a,ai)*(self.fplus(xi) - self.mu2(xi)*(1.5*self.Om(xi))**2/self.fplus(xi))*self.D(xi)**2/self.D(x)**2 / self.C(xi)
        return self.fplus(xi)*self.D(xi)**2*self.G2d(a,ai)/self.D(x)**2 / self.C(xi)
    def I1t(self, ai, a):
        x = log(a)
        xi = log(ai)
        if self.EFTDE: return (self.G1t(a,ai)*self.fplus(xi) + self.G2t(a,ai)*self.mu2(xi)*(1.5*self.Om(xi))**2/self.fplus(xi))*self.D(xi)**2/self.D(x)**2 / self.C(xi)
        return self.fplus(xi)*self.D(xi)**2*self.G1t(a,ai)/self.D(x)**2 / self.C(xi)
    def I2t(self, ai, a):
        x = log(a)
        xi = log(ai)
        if self.EFTDE: return self.G2t(a,ai)*(self.fplus(xi) - self.mu2(xi)*(1.5*self.Om(xi))**2/self.fplus(xi))*self.D(xi)**2/self.D(x)**2 / self.C(xi)
        return self.fplus(xi)*self.D(xi)**2*self.G2t(a,ai)/self.D(x)**2 / self.C(xi)

    # second order time integrals
    def mG1d(self, a):
        if self.vectorize:
            return quad_vec(self.I1d,self.lo,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
        else:
            return quad(self.I1d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mG2d(self, a):
        if self.vectorize:
            return quad_vec(self.I2d,self.lo,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
        else:
            return quad(self.I2d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mG1t(self, a):
        if self.vectorize:
            return quad_vec(self.I1t,self.lo,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
        else:
            return quad(self.I1t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mG2t(self, a):
        if self.vectorize:
            return quad_vec(self.I2t,self.lo,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
        else:
            return quad(self.I2t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]

    # quintessence time function
    def G(self, a):
        return self.mG1d(a) + self.mG2d(a)

    # third order coefficients
    def IU1d(self, ai, a):
        x = log(a)
        xi = log(ai)
        if self.EFTDE: return (self.G1d(a,ai)*self.fplus(xi)*self.mG1d(ai) + self.G2d(a,ai)*(1.5*self.Om(xi))**2*(self.mu2(xi)*self.mG1d(ai) + 0.5*self.mu22(xi)*1.5*self.Om(xi))/self.fplus(xi))*(self.D(xi)/self.D(x))**3 / self.C(xi)
        return self.fplus(xi)*self.mG1d(ai)*self.G1d(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IU2d(self, ai, a):
        x = log(a)
        xi = log(ai)
        if self.EFTDE: return ( self.G1d(a,ai)*self.fplus(xi)*self.mG2d(ai) + self.G2d(a,ai)*(1.5*self.Om(xi))**2*(self.mu2(xi)*self.mG2d(ai) - 0.5*self.mu22(xi)*1.5*self.Om(xi) )/self.fplus(xi) )*(self.D(xi)/self.D(x))**3 / self.C(xi)
        return self.fplus(xi)*self.mG2d(ai)*self.G1d(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IU1t(self, ai, a):
        x = log(a)
        xi = log(ai)
        if self.EFTDE: return (self.G1t(a,ai)*self.fplus(xi)*self.mG1d(ai) + self.G2t(a,ai)*(1.5*self.Om(xi))**2*(self.mu2(xi)*self.mG1d(ai) + 0.5*self.mu22(xi)*1.5*self.Om(xi))/self.fplus(xi))*(self.D(xi)/self.D(x))**3 / self.C(xi)
        return self.fplus(xi)*self.mG1d(ai)*self.G1t(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IU2t(self, ai, a):
        x = log(a)
        xi = log(ai)
        if self.EFTDE: return ( self.G1t(a,ai)*self.fplus(xi)*self.mG2d(ai) + self.G2t(a,ai)*(1.5*self.Om(xi))**2*( self.mu2(xi)*self.mG2d(ai) - 0.5*self.mu22(xi)*1.5*self.Om(xi) )/self.fplus(xi) )*(self.D(xi)/self.D(x))**3 / self.C(xi)
        return self.fplus(xi)*self.mG2d(ai)*self.G1t(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)

    def IV11d(self, ai, a):
        x = log(a)
        xi = log(ai)
        if self.EFTDE: return (self.G1d(a,ai)*self.fplus(xi)*self.mG1t(ai) + self.G2d(a,ai)*(1.5*self.Om(xi))**2*(self.mu2(xi)*self.mG1d(ai) + 0.5*self.mu22(xi)*1.5*self.Om(xi))/self.fplus(xi))*(self.D(xi)/self.D(x))**3 / self.C(xi)
        return self.fplus(xi)*self.mG1t(ai)*self.G1d(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IV12d(self, ai, a):
        x = log(a)
        xi = log(ai)
        if self.EFTDE: return self.G2d(a,ai)*(self.fplus(xi)*self.mG1t(ai) - (1.5*self.Om(xi))**2*(self.mu2(xi)*self.mG1d(ai) + 0.5*self.mu22(xi)*1.5*self.Om(xi))/self.fplus(xi))*(self.D(xi)/self.D(x))**3 / self.C(xi)
        return self.fplus(xi)*self.mG1t(ai)*self.G2d(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IV21d(self, ai, a):
        x = log(a)
        xi = log(ai)
        if self.EFTDE: return (self.G1d(a,ai)*self.fplus(xi)*self.mG2t(ai) + self.G2d(a,ai)*(1.5*self.Om(xi))**2*(self.mu2(xi)*self.mG2d(ai) - 0.5*self.mu22(xi)*1.5*self.Om(xi))/self.fplus(xi))*(self.D(xi)/self.D(x))**3 / self.C(xi)
        return self.fplus(xi)*self.mG2t(ai)*self.G1d(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IV22d(self, ai, a):
        x = log(a)
        xi = log(ai)
        if self.EFTDE: return self.G2d(a,ai)*(self.fplus(xi)*self.mG2t(ai) - (1.5*self.Om(xi))**2*(self.mu2(xi)*self.mG2d(ai) - 0.5*self.mu22(xi)*1.5*self.Om(xi))/self.fplus(xi))*(self.D(xi)/self.D(x))**3 / self.C(xi)
        return self.fplus(xi)*self.mG2t(ai)*self.G2d(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)

    def IV11t(self, ai,a):
        x = log(a)
        xi = log(ai)
        if self.EFTDE: return (self.G1t(a,ai)*self.fplus(xi)*self.mG1t(ai) + self.G2t(a,ai)*(1.5*self.Om(xi))**2*(self.mu2(xi)*self.mG1d(ai) + 0.5*self.mu22(xi)*1.5*self.Om(xi))/self.fplus(xi))*(self.D(xi)/self.D(x))**3 / self.C(xi)
        return self.fplus(xi)*self.mG1t(ai)*self.G1t(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IV12t(self, ai,a):
        x = log(a)
        xi = log(ai)
        if self.EFTDE: return self.G2t(a,ai)*(self.fplus(xi)*self.mG1t(ai) - (1.5*self.Om(xi))**2*(self.mu2(xi)*self.mG1d(ai) + 0.5*self.mu22(xi)*1.5*self.Om(xi))/self.fplus(xi))*(self.D(xi)/self.D(x))**3 / self.C(xi)
        return self.fplus(xi)*self.mG1t(ai)*self.G2t(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IV21t(self, ai,a):
        x = log(a)
        xi = log(ai)
        if self.EFTDE: return (self.G1t(a,ai)*self.fplus(xi)*self.mG2t(ai) + self.G2t(a,ai)*(1.5*self.Om(xi))**2*(self.mu2(xi)*self.mG2d(ai) - 0.5*self.mu22(xi)*1.5*self.Om(xi))/self.fplus(xi))*(self.D(xi)/self.D(x))**3 / self.C(xi)
        return self.fplus(xi)*self.mG2t(ai)*self.G1t(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IV22t(self, ai,a):
        x = log(a)
        xi = log(ai)
        if self.EFTDE: return self.G2t(a,ai)*(self.fplus(xi)*self.mG2t(ai) - (1.5*self.Om(xi))**2*(self.mu2(xi)*self.mG2d(ai) - 0.5*self.mu22(xi)*1.5*self.Om(xi))/self.fplus(xi))*(self.D(xi)/self.D(x))**3 / self.C(xi)
        return self.fplus(xi)*self.mG2t(ai)*self.G2t(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
   
    # third order time integrals
    def mU1d(self, a):
        if self.vectorize:
            return quad_vec(self.IU1d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]

        else:
            return quad(self.IU1d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mU2d(self, a):
        if self.vectorize:
            return quad_vec(self.IU2d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IU2d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mU1t(self, a):
        if self.vectorize:
            return quad_vec(self.IU1t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IU1t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mU2t(self, a):
        if self.vectorize:
            return quad_vec(self.IU2t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IU2t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]

    def mV11d(self, a):
        if self.vectorize:
            return quad_vec(self.IV11d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV11d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mV12d(self, a):
        if self.vectorize:
            return quad_vec(self.IV12d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV12d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mV21d(self, a):
        if self.vectorize:
            return quad_vec(self.IV21d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV21d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mV22d(self, a):
        if self.vectorize:
            return quad_vec(self.IV22d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV22d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]

    def mV11t(self, a):
        if self.vectorize:
            return quad_vec(self.IV11t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV11t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mV12t(self, a):
        if self.vectorize:
            return quad_vec(self.IV12t,self.lo,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-5)[0]
        else:
            return quad(self.IV12t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mV21t(self, a):
        if self.vectorize:
            return quad_vec(self.IV21t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV21t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mV22t(self, a):
        if self.vectorize:
            return quad_vec(self.IV22t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV22t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]

    def Y(self, a):
        if self.quintessence: return -3/14.*self.G(a)**2 + self.mV11d(a) + self.mV12d(a)
        else: return -3/14. + self.mV11d(a) + self.mV12d(a)


# ===========================================

    # #greens functions
    # def G1d(self, x, xi):
    #     ai = exp(xi)
    #     return(self.DDminus(xi)*self.D(x)-self.DD(xi)*self.Dminus(x))/(ai*self.W(xi))
    # def G2d(self, x, xi):
    #     ai = exp(xi)
    #     return self.fplus(xi)*(self.Dminus(x)*self.D(xi)-self.D(x)*self.Dminus(xi))/(ai*ai*self.W(xi))
    # def G1t(self, x, xi):
    #     a = exp(x)
    #     ai = exp(xi)
    #     return a*(self.DDminus(xi)*self.DD(x)-self.DD(xi)*self.DDminus(x))/(self.fplus(x)*ai*self.W(xi))
    # def G2t(self, x, xi):
    #     a = exp(x)
    #     ai = exp(xi)
    #     return a*self.fplus(xi)*(self.DDminus(x)*self.D(xi)-self.DD(x)*self.Dminus(xi))/(self.fplus(x)*ai*ai*self.W(xi))

    # # second order coefficients
    # # the last exp(x) comes from the change of integral variable
    # def I1d(self, xi, x):
    #     return self.fplus(xi)*self.D(xi)**2*self.G1d(x,xi)/self.D(x)**2 / self.C(xi)*exp(xi)
    # def I2d(self, xi, x):
    #     return self.fplus(xi)*self.D(xi)**2*self.G2d(x,xi)/self.D(x)**2 / self.C(xi)*exp(xi)
    # def I1t(self, xi, x):
    #     return self.fplus(xi)*self.D(xi)**2*self.G1t(x,xi)/self.D(x)**2 / self.C(xi)*exp(xi)
    # def I2t(self, xi, x):
    #     return self.fplus(xi)*self.D(xi)**2*self.G2t(x,xi)/self.D(x)**2 / self.C(xi)*exp(xi)

    # # second order time integrals
    # def mG1d(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.I1d,lo,x,args=(x,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
    #     else:
    #         return quad(self.I1d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mG2d(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.I2d,lo,x,args=(x,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
    #     else:
    #         return quad(self.I2d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mG1t(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.I1t,lo,x,args=(x,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
    #     else:
    #         return quad(self.I1t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mG2t(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.I2t,lo,x,args=(x,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
    #     else:
    #         return quad(self.I2t,lo,x,args=(x,), epsrel=self.epsrel)[0]

    # # quintessence time function
    # def G(self, x):
    #     return self.mG1d(x) + self.mG2d(x)

    # # third order coefficients
    # def IU1d(self, xi, x):
    #     return self.fplus(xi)*self.mG1d(xi)*self.G1d(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IU2d(self, xi, x):
    #     return self.fplus(xi)*self.mG2d(xi)*self.G1d(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IU1t(self, xi, x):
    #     return self.fplus(xi)*self.mG1d(xi)*self.G1t(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IU2t(self, xi, x):
    #     return self.fplus(xi)*self.mG2d(xi)*self.G1t(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)

    # def IV11d(self, xi, x):
    #     return self.fplus(xi)*self.mG1t(xi)*self.G1d(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IV12d(self, xi, x):
    #     return self.fplus(xi)*self.mG1t(xi)*self.G2d(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IV21d(self, xi, x):
    #     return self.fplus(xi)*self.mG2t(xi)*self.G1d(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IV22d(self, xi, x):
    #     return self.fplus(xi)*self.mG2t(xi)*self.G2d(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)

    # def IV11t(self, ai,a):
    #     return self.fplus(xi)*self.mG1t(xi)*self.G1t(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IV12t(self, ai,a):
    #     return self.fplus(xi)*self.mG1t(xi)*self.G2t(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IV21t(self, ai,a):
    #     return self.fplus(xi)*self.mG2t(xi)*self.G1t(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IV22t(self, ai,a):
    #     return self.fplus(xi)*self.mG2t(xi)*self.G2t(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
   
    # # third order time integrals
    # def mU1d(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IU1d,lo,x,args=(x,), epsrel=self.epsrel)[0]

    #     else:
    #         return quad(self.IU1d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mU2d(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IU2d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IU2d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mU1t(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IU1t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IU1t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mU2t(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IU2t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IU2t,lo,x,args=(x,), epsrel=self.epsrel)[0]

    # def mV11d(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IV11d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IV11d,lo,x,args=(a,), epsrel=self.epsrel)[0]
    # def mV12d(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IV12d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IV12d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mV21d(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IV21d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IV21d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mV22d(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IV22d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IV22d,lo,x,args=(x,), epsrel=self.epsrel)[0]

    # def mV11t(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IV11t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IV11t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mV12t(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IV12t,lo,x,args=(x,), epsrel=self.epsrel, epsabs=1.49e-5)[0]
    #     else:
    #         return quad(self.IV12t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mV21t(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IV21t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IV21t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mV22t(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IV22t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IV22t,lo,x,args=(x,), epsrel=self.epsrel)[0]

    # def Y(self, x):
    #     if self.quintessence: return -3/14.*self.G(x)**2 + self.mV11d(x) + self.mV12d(x)
    #     else: return -3/14. + self.mV11d(x) + self.mV12d(x)
import numpy as np
import inspect
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm


class BPModel:
    def __init__(self):
        super().__init__()
        self._xmax = 20e-6
        self._ymax = 20e-6
        self._N = 256
        self.update_grids()
        self.RI = None
        self.lambda0 = 1000e-9
        self.field = None
        self.n0 = 1
        self._Lz = 1e-3
        self.update_dz()
        self.build_absorption_profile()

    @property
    def Lz(self):
        return self._Lz
    
    @Lz.setter
    def Lz(self, val):
        self._Lz = val
        self.update_dz()

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, val):
        self._N = val
        self.update_grids()

    @property
    def xmax(self):
        return self._xmax

    @xmax.setter
    def xmax(self, val):
        self._xmax = val
        self.update_grids()

    @property
    def ymax(self):
        return self._ymax

    @ymax.setter
    def ymax(self, val):
        self._ymax = val
        self.update_grids()

    def update_grids(self):
        self.x = np.linspace(-self._xmax/2, self._xmax/2, self._N)
        self.y = np.linspace(-self._ymax/2, self._ymax/2, self._N)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

    def update_dz(self):
        self.dz = 1e-6

    def set_RI_from_function(self, func):
        numArgs = len(inspect.signature(func).parameters)
        if numArgs == 3:
            self.RI = func(self.X, self.Y, self)
        elif numArgs == 2:
            self.RI = func(self.X, self.Y)
        else:
            raise ValueError("The RI function must take 2 or 3 parameters: X, Y and BPModel (optionnal)")
    
    def solve_modes(self, Nmodes=10, plotModes=False, coreRadius=5e-6):
        h = max(self.x[1] - self.x[0], self.y[1] - self.y[0])
        k0 = 2 * np.pi / self.lambda0
        N = self.N 

        lowerdiag = np.ones(N**2)
        lowerdiag[N-1::N] = 0 
        upperdiag = np.roll(lowerdiag, 1)

        diagonals = [
            (-4 / h**2) * np.ones(N**2),
            (1 / h**2) * lowerdiag,
            (1 / h**2) * upperdiag,
            (1 / h**2) * np.ones(N**2),
            (1 / h**2) * np.ones(N**2)
        ]
        offsets = [0, -1, 1, -N, N]

        Lap = sp.diags(diagonals, offsets, format='csr')
        Op = Lap + sp.diags((k0**2 * self.RI.ravel()**2), 0)

        sigma = (np.max(self.RI) * k0)**2 

        vals, vecs = spla.eigs(Op, k=Nmodes, sigma=sigma, which='LM')
        neff = np.sqrt(np.real(vals)) / k0
        modes = vecs.reshape(self.N, self.N, Nmodes)

        self.modes = modes
        self.neff = neff

        if plotModes:
            for i in range(Nmodes):
                I = np.abs(modes[:, :, i])**2
                I /= I.max()
                plt.figure()
                plt.imshow(I, extent=[self.x[0]*1e6, self.x[-1]*1e6,
                                    self.y[0]*1e6, self.y[-1]*1e6], origin='lower', cmap='jet')
                plt.title(f"Mode {i+1}, neff = {neff[i]:.6f}")
                plt.colorbar(label="Intensity")
                plt.xlabel("x (µm)")
                plt.ylabel("y (µm)")
                plt.show()
    
    def set_initial_field_from_mode(self, mode_index=0, combination=False):
        if not hasattr(self, 'modes'):
            raise RuntimeError("Call solve_modes() before selecting a mode.")

        Nmodes_total = self.modes.shape[2]

        if combination:
            if mode_index == 'all':
                indices = np.arange(Nmodes_total)
            elif isinstance(mode_index, (list, np.ndarray)):
                indices = np.array(mode_index)
                if np.any((indices < 0) | (indices >= Nmodes_total)):
                    raise ValueError(f"Modes idexes should be in [0, {Nmodes_total-1}]")
            else:
                raise ValueError("In combination mode, mode_index should be 'all' or a list of indexes")

            coeffs = np.random.randn(len(indices)) + 1j * np.random.randn(len(indices))
            selected_modes = self.modes[:, :, indices]
            field = np.sum(selected_modes * coeffs[None, None, :], axis=2)
        else:
            if not isinstance(mode_index, int):
                raise ValueError("In standard mode, mode_index should be an integer")
            if mode_index < 0 or mode_index >= Nmodes_total:
                raise ValueError(f"mode_index should be in [0, {Nmodes_total-1}]")
            field = self.modes[:, :, mode_index]

        self.field = field

    
    def show_field(self):
        if not hasattr(self, 'field'):
            raise RuntimeError("No field has been defined")
        if self.RI is None:
            raise RuntimeError("No refractive index profile has been defined")
        
        intensity = np.abs(self.field)**2
        intensity /= intensity.max()  

        plt.figure()
        plt.imshow(intensity.T, extent=[self.x[0]*1e6, self.x[-1]*1e6, self.y[0]*1e6, self.y[-1]*1e6], 
                origin='lower', cmap='jet')
        # Ajouter les contours de RI
        levels = np.linspace(np.min(self.RI), np.max(self.RI), 10)
        plt.contour(self.x*1e6, self.y*1e6, self.RI.T, levels=levels, colors='white', linewidths=0.8)
        plt.xlabel('x (µm)')
        plt.ylabel('y (µm)')
        plt.title('Field intensity')
        plt.show()

    def propagate_full(self, animate=True):
        self.construct_operators()

        steps = int(np.ceil(self.Lz / self.dz))

        if animate:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            im1 = ax1.imshow(np.abs(self.field)**2, extent=[self.x[0]*1e6, self.x[-1]*1e6,
                                                        self.y[0]*1e6, self.y[-1]*1e6],
                            origin='lower', cmap='jet', vmin=0, vmax=1)
            ax1.set_title("Intensity |Ez|²")
            ax1.set_xlabel("x (µm)")
            ax1.set_ylabel("y (µm)")
            fig.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(np.angle(self.field), extent=[self.x[0]*1e6, self.x[-1]*1e6,
                                                        self.y[0]*1e6, self.y[-1]*1e6],
                            origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi)
            ax2.set_title("Phase arg(Ez)")
            ax2.set_xlabel("x (µm)")
            ax2.set_ylabel("y (µm)")
            fig.colorbar(im2, ax=ax2)

            def update(frame):

                Ez_vec = spla.spsolve(self.B, self.C @ self.field.ravel())

                Ez = Ez_vec.reshape(self._N, self._N)

                self.field = Ez

                im1.set_data(np.abs(Ez)**2 / np.abs(Ez).max()**2)
                im2.set_data(np.angle(Ez))
                return im1, im2

            ani = animation.FuncAnimation(fig, update, frames=steps, interval=2, blit=True)
            plt.show()
        else:
            for _ in tqdm(range(steps), desc="Propagation distance", unit="Steps", colour='green'):
                Ez_vec = spla.spsolve(self.B, self.C @ self.field.ravel())
                Ez = Ez_vec.reshape(self._N, self._N)
                self.field = Ez 
            self.show_field()

    def build_absorption_profile(self, alpha_max=3e14, border_ratio=0.5):
  
        Xnorm = 2 * np.abs(self.X) / self._xmax
        Ynorm = 2 * np.abs(self.Y) / self._ymax

        ramp_x = np.clip((Xnorm - (1 - border_ratio)) / border_ratio, 0, 1)
        ramp_y = np.clip((Ynorm - (1 - border_ratio)) / border_ratio, 0, 1)

        alpha = alpha_max * (ramp_x + ramp_y)**2
        self.abs_profile = np.exp(-alpha * self.dz)

    def construct_operators(self) :
        dx = self._xmax/self._N
        dy = self._ymax/self._N
        k0 = 2*np.pi/self.lambda0
        ax = self.dz/(2*dx**2)
        ay = self.dz/(2*dy**2)
        b = 2*1j*k0*self.n0 + (self.dz/(dx**2)) + (self.dz/(dy**2)) - ((k0**2*self.dz)/2)*(self.RI.ravel()**2 - self.n0**2)
        c = 2*1j*k0*self.n0 - (self.dz/(dx**2)) - (self.dz/(dy**2)) + ((k0**2*self.dz)/2)*(self.RI.ravel()**2 - self.n0**2)

        Bindiag = -ax*np.ones(self.N**2-1)
        Boutdiag = -ay*np.ones(self.N**2-self.N)
        for i in range(1, self.N):
            Bindiag[i * self.N - 1] = 0
        
        self.B = sp.diags([b], [0], format='csr') + sp.diags([Bindiag,Bindiag], offsets=[-1,1], format='csr')+ sp.diags([Boutdiag,Boutdiag], offsets=[-self.N,self.N], format='csr')
        self.C = sp.diags([c], [0], format='csr') + sp.diags([-Bindiag,-Bindiag], offsets=[-1,1], format='csr')+ sp.diags([-Boutdiag,-Boutdiag], offsets=[-self.N,self.N], format='csr')
    
    def shift_field(self, dx=0, dy=0):

        if self.field is None:
            raise RuntimeError("No field defined")

        N = self.N
        fx = int(np.round(dx))
        fy = int(np.round(dy))

        shifted = np.zeros_like(self.field, dtype=complex)

        x_start_src = max(0, -fx)
        x_end_src = N - max(0, fx)
        y_start_src = max(0, -fy)
        y_end_src = N - max(0, fy)

        x_start_dst = max(0, fx)
        x_end_dst = N - max(0, -fx)
        y_start_dst = max(0, fy)
        y_end_dst = N - max(0, -fy)

        
        shifted[x_start_dst:x_end_dst, y_start_dst:y_end_dst] = \
            self.field[x_start_src:x_end_src, y_start_src:y_end_src]

        self.field = shifted

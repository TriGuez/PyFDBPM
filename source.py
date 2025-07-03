import numpy as np
import inspect
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from tqdm import tqdm

plt.style.use('dark_background')

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
        levels = np.linspace(np.min(self.RI), np.max(self.RI), 10)
        #plt.contour(self.x*1e6, self.y*1e6, self.RI.T, levels=levels, colors='white', linewidths=0.8)
        plt.xlabel('x (µm)')
        plt.ylabel('y (µm)')
        plt.title('Field intensity')
        plt.show()

   
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

    def operators_DGAI(self) :
        dx = self._xmax/self._N
        dy = self._ymax/self._N
        self.k0 = 2*np.pi/self.lambda0
        ax = self.dz/(4*1j*self.k0*self.n0*dx**2)
        ay = self.dz/(4*1j*self.k0*self.n0*dy**2)
        d = 1-2*ax-2*ay
        diag0P1 = 2*ax*np.ones(self._N**2)
        diag1P1 = -ax*np.ones(self._N**2-1)
        for i in range(1, self._N):
            diag0P1[i * self._N - 1] = 0
            diag1P1[i * self._N - 1] = 0
        diag1P1 = -ax*np.ones(self._N**2-1)
        self.P1 = sp.diags([2*ax*np.ones(self._N**2)], [0], format='csr') + sp.diags([-ax*np.ones(self._N**2-1),-ax*np.ones(self._N**2-1)], [-1,1], format='csr')
        self.P2 = sp.diags([2*ay*np.ones(self._N**2)], [0], format='csr') + sp.diags([-ay*np.ones(self._N**2-self._N),-ay*np.ones(self._N**2-self._N)], [-self._N,self._N], format='csr')
        self.Q = sp.diags([d*np.ones(self._N**2)], [0], format='csr') + sp.diags([ax*np.ones(self._N**2-1),ax*np.ones(self._N**2-1)], [-1,1], format='csr') + sp.diags([-ay*np.ones(self._N**2-self._N),-ay*np.ones(self._N**2-self._N)], [-self._N,self._N], format='csr')
        
    
    def propagate_full(self, animate=True):
        self.operators_DGAI()
        LU1 = spla.splu((sp.eye(self._N**2)+self.P1).tocsc())
        LU2 = spla.splu((sp.eye(self._N**2)+self.P2).tocsc())
        steps = int(np.ceil(self.Lz / self.dz))
        absorption = self.calculate_absorption()
        self.initPower = np.abs(np.trapz(np.trapz(self.field * self.field.conjugate(),self.x), self.y))**2

        if animate:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            
            im1 = ax1.imshow(
                np.abs(self.field)**2,
                extent=[self.x[0]*1e6, self.x[-1]*1e6, self.y[0]*1e6, self.y[-1]*1e6],
                origin='lower', cmap='jet', vmin=0, vmax=1
            )
            ax1.set_title("Intensity |Ez|²")
            ax1.set_xlabel("x (µm)")
            ax1.set_ylabel("y (µm)")
            fig.colorbar(im1, ax=ax1)

            
            phase = np.angle(self.field)
            phase_norm = (phase + np.pi) / (2 * np.pi)
            intensity = np.abs(self.field)
            maxE0 = intensity.max()
            alpha = np.maximum(0, (1 + np.log10((intensity / maxE0)**2) / 3))
            cmap = cm.get_cmap('twilight')
            rgba_img = cmap(phase_norm)
            rgba_img[..., 3] = alpha

            im2 = ax2.imshow(
                rgba_img,
                extent=[self.x[0]*1e6, self.x[-1]*1e6, self.y[0]*1e6, self.y[-1]*1e6],
                origin='lower', vmin=0, vmax=1
            )
            ax2.set_title("Phase arg(Ez)")
            ax2.set_xlabel("x (µm)")
            ax2.set_ylabel("y (µm)")
            fig.colorbar(cm.ScalarMappable(cmap='twilight', norm=plt.Normalize(-np.pi, np.pi)), ax=ax2)

            
            distance_text = ax1.text(
                0.02, 0.95, '', transform=ax1.transAxes, color='white',
                fontsize=14, bbox=dict(facecolor='black', alpha=0.5)
            )

            steps = int(self.Lz / self.dz)

            def update(frame):
                
                rhs1 = -self.P2 @ self.field.ravel() + self.Q @ self.field.ravel()
                Ezprime = LU1.solve(rhs1)
                rhs2 = Ezprime + self.P2 @ self.field.ravel()
                Ez_vec = LU2.solve(rhs2)
                Ez = Ez_vec.reshape(self._N, self._N)
                phase_factor = np.exp(-self.dz*self.k0/(2*1j*self.n0)*(self.RI**2 - self.n0**2))
                self.field = Ez * (phase_factor * absorption)

                im1.set_data(np.abs(self.field)**2 / np.abs(self.field).max()**2)

                intensity = np.abs(self.field)
                maxE0 = intensity.max()
                alpha = np.maximum(0, (1 + np.log10((intensity / maxE0)**2) / 3))
                phase = np.angle(self.field)
                phase_norm = (phase + np.pi) / (2 * np.pi)
                rgba_img = cmap(phase_norm)
                rgba_img[..., 3] = alpha
                im2.set_data(rgba_img)

                distance = (frame+1) * self.dz * 1e6  
                distance_text.set_text(f"z = {distance:.1f} µm")

                return im1, im2, distance_text

            ani = animation.FuncAnimation(
                fig, update, frames=steps, interval=2, blit=True, repeat=False
            )
            plt.show()


        else:
            for _ in tqdm(range(steps), desc="Propagation distance", unit="Steps", colour='green'):
                rhs1 = -self.P2 @ self.field.ravel() + self.Q @ self.field.ravel()
                Ezprime = LU1.solve(rhs1)
                rhs2 = Ezprime + self.P2 @ self.field.ravel()
                Ez_vec = LU2.solve(rhs2)
                Ez = Ez_vec.reshape(self._N, self._N)
                phase = np.exp(-self.dz*self.k0/(2*1j*self.n0)*(self.RI**2 - self.n0**2))
                self.field = Ez * (phase * absorption)
                self.remainingPower = np.abs(np.trapz(np.trapz(self.field * self.field.conjugate(),self.x), self.y))**2
                self.field /= np.sqrt(np.sum(np.abs(self.field)**2))
            self.show_field()
        
    def calculate_absorption(self):
        alpha = 3e14  
        xEdge = self.x.max() * 0.8  
        yEdge = self.y.max() * 0.8  
        
        dist_to_edge = np.maximum(
            np.abs(self.X) - xEdge,
            np.abs(self.Y) - yEdge)
        
        absorption = np.ones_like(self.X, dtype=complex)
        mask = dist_to_edge > 0
        absorption[mask] = np.exp(-self.dz *alpha * dist_to_edge[mask]**2)
        
        return absorption

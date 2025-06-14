import numpy as np
import inspect
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

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
        N = self.N  # suppose Nx == Ny ici

        # Laplacian operator (5-point stencil)
        lowerdiag = np.ones(N**2)
        lowerdiag[N-1::N] = 0  # remove wrap connections
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
            # Cas : tous les modes sont combinés
            if mode_index == 'all':
                indices = np.arange(Nmodes_total)
            # Cas : liste d’indices fournie
            elif isinstance(mode_index, (list, np.ndarray)):
                indices = np.array(mode_index)
                if np.any((indices < 0) | (indices >= Nmodes_total)):
                    raise ValueError(f"Tous les indices doivent être dans [0, {Nmodes_total-1}]")
            else:
                raise ValueError("En mode combinaison, mode_index doit être 'all' ou une liste d'indices.")

            coeffs = np.random.randn(len(indices)) + 1j * np.random.randn(len(indices))
            selected_modes = self.modes[:, :, indices]
            field = np.sum(selected_modes * coeffs[None, None, :], axis=2)
        else:
            if not isinstance(mode_index, int):
                raise ValueError("En mode simple, mode_index doit être un entier.")
            if mode_index < 0 or mode_index >= Nmodes_total:
                raise ValueError(f"mode_index doit être dans [0, {Nmodes_total-1}]")
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


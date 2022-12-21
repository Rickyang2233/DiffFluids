import numpy as np
from tqdm import tqdm

import phi.torch.flow as flow

class SmokeSimulator:
    def __init__(self, 
                 resolution: tuple,
                 time_step: float,
                 source_locations: list,
                 source_radius: int=2,
                 source_strength: float=1.0
                 ):
        flow.TORCH.set_default_device('GPU')
        self.res = resolution
        self.dt = time_step
        self.src_locs = source_locations
        self.src_rad = source_radius
        self.src_str = source_strength
        self.inflow = self._get_inflow()
    
    def _get_inflow(self) -> flow.CenteredGrid:
        inflow_loc = flow.tensor(self.src_locs, 
                                flow.batch('inflow_loc'), 
                                flow.channel(vector='x, y'))
        inflow = self.src_str * flow.CenteredGrid(
            flow.Sphere(center=inflow_loc, radius=self.src_rad),
            flow.extrapolation.BOUNDARY, 
            x=self.res[0], 
            y=self.res[1])
        return inflow
    
    def step(self, density: flow.CenteredGrid, velocity: flow.StaggeredGrid) -> tuple:
        # advect density
        density = flow.advect.semi_lagrangian(density, velocity, self.dt) + self.inflow * self.dt
        # calculate buoyancy
        buoyancy = (density * (0, 1)).at(velocity)
        # advect velocity
        velocity = flow.advect.semi_lagrangian(velocity, velocity, self.dt) + buoyancy * self.dt
        velocity, _ = flow.fluid.make_incompressible(velocity)
        return (density, velocity)
    
    def simulate(self, total_time: float, save_point: list) -> dict:
        density = flow.CenteredGrid(0, extrapolation=flow.extrapolation.BOUNDARY, x=self.res[0], y=self.res[1],
                                bounds=flow.Box(x=self.res[0], y=self.res[1]))
        velocity = flow.StaggeredGrid(0, extrapolation=flow.extrapolation.ZERO, x=self.res[0], y=self.res[1],
                                    bounds=flow.Box(x=self.res[0], y=self.res[1]))
        tmp = []
        for t in tqdm(np.arange(0, total_time+self.dt, self.dt)):
            density, velocity = self.step(density, velocity)
            if t in save_point:
                tmp.append((density, velocity))
        return dict(zip(save_point, tmp))
        
# 01.02-Propagating-Uncertainty-3D-Lorenz-Attractor

This experiment evaluates GP-LSTM's (Gaussian Process regression in combination with LSTM's) on their ability to forecast the predictive distribution of dynamical systems.
The GP-LSTM models are built using the keras-gp library (https://github.com/alshedivat/keras-gp) with an octave engine.

Please check the [README_Lorenz_attractor File](README_Lorenz_attractor.docx) for detailed instructions on how to run the experiment.

*01.02 3-Dimensional Lorenz Attractor*

The predictive distributions of the GP-LSTM model are evaluated on the 3-Dimensional Lorenz Attractor.

This attractor system was initially developed by Edward N. Lorenz as a simple
model for atmospheric convection and dissipative hydrodynamic flows. The
system exhibits chaotic behavior for certain parameter values.
An attractor describes a collection of numerical values to which a dynamical
system tends to evolve. The set of values can be a single point, a curve or a
highly complex shape known as a strange attractor. For points near or within
the attractor space, the values stay close even if disturbed. The points follow the
trajectories of the attractor.

The Lorenz attractor consists of a system of three differential equations describing
the temperature and convection dynamics of a two dimensional fluid layer.

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial x}{\partial t} \ = \sigma(y-x)">

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial y}{\partial t} \ = x(\rho-z)-y">

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial z}{\partial t} \ = xy-\beta z">

Where:

• x proportional to the rate of convection

• y proportional to the horizontal temperature variation

• z proportional to the vertical temperature variation

• σ, ρ, β: System parameter proportional to the Prandtl number, Rayleigh
number and to physical dimensions of the fluid layer itself

Applying the above described system of differential equations results in the plot below:

<img src="./Figures/Lorenz3D.jpg"/>

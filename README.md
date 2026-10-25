# Spaceship-Trajectory-Calculation

Project Overview and Mission Objectives

The objective of this project is to simulate and analyze the interplanetary trajectory of a spaceship traveling from Earth to Mars while accounting for the gravitational influences of the Sun and other celestial bodies.

In this project, we compute the spacecraft trajectory using multiple numerical integration solvers, such as the Runge-Kutta and Leapfrog methods, and demonstrate why simpler integration schemes like the Euler method fail to maintain stability and accuracy over long-term orbital simulations. To ensure improved numerical stability and proper scaling across the simulation, the equations of motion are formulated and solved using dimensionless, rescaled parameters.

A core part of our task is route optimization. We optimize the initial launch velocity and launch angle so that the spacecraft accurately intercepts the orbit of Mars, applying Hohmann transfer orbits or similar orbital mechanics techniques to model realistic interplanetary transfers. Alongside the trajectory calculations, we perform a rigorous error analysis to quantify and visualize integration errors over time, comparing the accuracy and computational performance of the different numerical solvers.

Finally, we present the results through static 2D or 3D trajectory plots showing the path of the spaceship relative to the Sun, Earth, and Mars, as well as a dynamic animation of the complete mission. This visualization illustrates the orbital departure from Earth, the interplanetary transfer trajectory, and the approach to the orbit of Mars, while highlighting critical mission events including propulsion burns, gravity assists, and orbital capture.
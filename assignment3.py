import numpy as np
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['OBSTACLES'] = True

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 15
        self.dt = 0.2

        # Reference or set point the controller will achieve.
        self.reference1 = [10, 0, 0]
        self.reference2 = None

        self.x_obs = 5
        self.y_obs = 0.1

    def plant_model(self,prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3]

        a_t = pedal
        x_t = x_t + np.cos(psi_t) * v_t * dt
        y_t = y_t + np.sin(psi_t) * v_t * dt
        v_t = v_t + a_t * dt - v_t/25
        psi_t = psi_t + v_t * (np.tan(steering)/2.5) * dt

        return [x_t, y_t, psi_t, v_t]

    def cost_function(self,u, *args):
        state = args[0]
        ref = args[1]
        cost = 0.0

        car_width = 1.0
        car_height = 2.5
        car_radius = 2.5
        object_radius = 0.5

        for i in range(0, self.horizon):
            state = self.plant_model(state, self.dt, u[i*2], u[i*2+1])

            # Distance cost
            distance_cost = np.sqrt( ((ref[0] - state[0]) ** 2) + ((ref[1] - state[1]) ** 2) )

            # Angle cost
            angle_cost = 0
            if distance_cost < 4:
                angle_cost = (ref[2] - state[2]) ** 2

            # Object cost
            obj_distance = (np.sqrt(((self.x_obs - state[0]) ** 2) + ((self.y_obs - state[1]) ** 2)))
            obj_cost = 0
            if obj_distance < 1.5:
                obj_cost = 1000/obj_distance

            cost += distance_cost + obj_cost + angle_cost

        return cost

sim_run(options, ModelPredictiveControl)

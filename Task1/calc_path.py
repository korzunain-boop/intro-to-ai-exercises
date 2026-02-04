import numpy as np

def calc_path(control: np.ndarray, time: int = 200, dt: float = 0.1):
    posx = 0.0
    posz = 0.0
    velx = 0.0
    velz = 0.0
    pathx = [posx]
    pathz = [posz]

    for t in range(time):
        if posz < 0:
            break

        cx, cz = control[t]
        velx += (cx * 15 - 0.5 * velx) * dt
        velz += (cz * 15 - 9.8 - 0.5 * velz) * dt
        posx += velx * dt
        posz += velz * dt

        pathx.append(posx)
        pathz.append(posz)

    return np.array(pathx), np.array(pathz)

def calc_target(control: np.ndarray):
    pathx, pathz = calc_path(control)
    target = -(pathx[-1] - 350) ** 2
    return target

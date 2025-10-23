# range of each finger parameter
# 1: [-0.873,0.873]
# 2: [-1.4,0.785]
# 3: [0,1,4]
# 4: [-0.524,1.48]
import numpy as np

fingers_min=np.array([-0.873,-1.4,0,-0.524])
fingers_max=np.array([0.873,0.785,1.4,1.48])

# fingers_min=np.array([-0.873,-1.4,0])
# fingers_max=np.array([0.873,0.785,1.4])

fingers_range=fingers_max-fingers_min

fingers_mean=(fingers_min+fingers_max)/2
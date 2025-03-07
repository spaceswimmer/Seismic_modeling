import numpy as np
import matplotlib.pyplot as plt


def model_creation():
    spacing = (0.5, 0.5)
    shape = (int(200//spacing[0]), int(114//spacing[1]))
    origin = (0, 0)
    nbl = 10
    so = 4

    vp = np.ones(shape, dtype=np.float32)
    rho = np.ones(shape, dtype=np.float32) * 1.6
    # vs примем просто 0.5 от vp

    #layer1
    l1 = int(35//spacing[1])
    vp[:,:l1] = 1.45
    #layer2
    l2 = int(70//spacing[1])
    vp[:,l1:l2] = 1.55
    #layer3
    l3 = int(114//spacing[1])
    vp[:,l2:l3] = 1.87
    
    plt.imshow(vp)
    plt.show()
def main():
    pass

if __name__ == "__main__":
    model_creation()
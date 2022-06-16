from main import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

if __name__ == '__main__':
    # script is obsolete, as KE_uniform_sampling uses the most suitable method (with threshold measured in this script)
    # for decoding the video
    beginrate = 0.1
    endrate = 3
    sampling_rates = np.linspace(beginrate, endrate, num=2)

    t_discarding = []
    t_framepick = []

    for rate in sampling_rates:
        t1 = time.time()
        keyframes_data, keyframe_indices, video_fps = fast_uniform_sampling(sys.argv[1], rate, 0.85)
        t_discarding.append(time.time()-t1)
        t2 = time.time()
        keyframes_data, keyframe_indices, video_fps = KE_uniform_sampling(sys.argv[1], rate, 0.85)
        t_framepick.append(time.time()-t2)

    sampling_rates_new = np.linspace(beginrate, endrate, num=50, endpoint=True)
    f_discarding = interp1d(sampling_rates, t_discarding)
    f_framepick = interp1d(sampling_rates, t_framepick)

    plt.plot(sampling_rates_new, f_discarding(sampling_rates_new), '-', label='Read all frames, retrieve by modulo')
    plt.plot(sampling_rates_new, f_framepick(sampling_rates_new), '-', label='CAP_PROP_POS at selected indices')
    plt.legend(loc='best')
    plt.xlabel("Sampling rate [FPS]")
    plt.ylabel("Time [s]")
    plt.title("Computation time decoding frames using uniform sampling")
    plt.show()


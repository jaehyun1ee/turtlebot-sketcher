import numpy as np
import matplotlib.pyplot as plt

log = open("./logs/monitor.csv", 'r')
lines = log.readlines()
lines.pop(0)
lines.pop(0)

avg_pixel_diff = []
sum_pixel_diff = 0
ep_count = 0
for line in lines:
    pixel_diff = float(line.strip().split(',')[4])
    sum_pixel_diff += pixel_diff
    ep_count += 1
    
    if ep_count == 50:
        avg_pixel_diff.append(sum_pixel_diff / ep_count)
        sum_pixel_diff = 0
        ep_count = 0

plt.plot([50 * (i + 1) for i in range(len(avg_pixel_diff))], avg_pixel_diff)
plt.xlabel("episodes")
plt.ylabel("pixels")
plt.ylim(0, 16)
plt.show()

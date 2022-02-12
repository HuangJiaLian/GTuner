import pyaudio
import numpy as np 
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt

# Convert degree to radian
def d2r(degree): return degree * np.pi / 180.

# Press 'q' to quit the program
def on_press(event):
    global END, in_stream, p
    if event.key == 'q':
        END = True
        plt.close()
        in_stream.stop_stream()
        in_stream.close()
        p.terminate()

#================================================================
# Notes_guitar = ['E2','A2','D3','G3','B3','E4']
Notes_guitar = ['E','A','D','G','B','E']
freq_guitar = np.array([82.4069, 110.0000, 146.8324,\
                        195.9977, 246.9417, 329.6276])
freq_ticks = np.array([0, 82.4069, 90, 110.0000, 146.8324, 180,\
                        195.9977, 246.9417, 270, 329.6276])
tick_notes = ['0 Hz','E','90','A','D','180','G','B','270','E']
#================================================================

FORMAT = pyaudio.paInt16 # 16 bit int
CHANNELS = 1 # Number of input channels
SAMPLE_RATE = 44100  # Sampling rate/frequency (Hz)
CHUNK = 1024*16*2 # Number of sample frames per buffer
SAMPLE_INTERVAL = 1/SAMPLE_RATE 
END = False # End flag

# Initio sound input
p = pyaudio.PyAudio()
in_stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,\
                input=True, frames_per_buffer=CHUNK)

# Draw panel
r_panel = 30
pointer_len = r_panel - 1
pointer_color = '#E7E0CD'
pointer_width = 1
spectrum_base = 15
divide_factor = 40
# mpl.rcParams['toolbar'] = 'None'
fig = plt.figure()
plt.rcParams["font.weight"] = "bold"
fig.patch.set_facecolor('#F7FBF8')
# fig.canvas.toolbar_visible = False
ax = plt.subplot(projection='polar')
ax.set_facecolor('#305996')
plt.get_current_fig_manager().set_window_title('GTuner')
fig.canvas.mpl_connect('key_press_event', on_press)
ax.set_xticks(d2r(freq_ticks)) 
ax.set(xticklabels=tick_notes)

ax.set_ylim(0,30)
ax.set_yticks([30]) 
ax.set(yticklabels=[])
# ax.spines['polar'].set_visible(False)
ax.spines['polar'].set_color('#305996')
ax.tick_params(axis='x', colors='#305996')
plt.grid()

scale = np.arange(0, 360, 10)
scale_end_r = r_panel
scale_start_r = r_panel - 1
scale_w_min = 0.7
scale_w_max = 2.0

# Center dot
ax.scatter(0, 0, c=pointer_color, s=32, cmap='hsv', alpha=1)
for s in scale:
    ax.vlines(d2r(s), scale_end_r, scale_start_r, colors=pointer_color,\
              linewidth= scale_w_min, zorder=3)
for f in freq_guitar:
    ax.vlines(d2r(f), scale_end_r, scale_start_r, colors=pointer_color,\
              linewidth= scale_w_max, zorder=3)



# Cut freq. of band pass filter
lowcut, highcut = 75.0, 1250.0
freq_range = [75, 350]
freq = np.fft.rfftfreq(CHUNK, d=1./SAMPLE_RATE)
mask = (freq < freq_range[0]) + (freq > freq_range[1])

mask_plot = freq < 360
freq_to_plot = freq[mask_plot]
line0, = ax.plot(d2r(freq_to_plot), 50*np.random.rand(len(freq_to_plot)),\
                 color=pointer_color, linewidth= pointer_width)

# Interreactive mode on
plt.ion()
plt.tight_layout()
plt.show()

while END==False:
    # Read binary data to buffer;
    # False: silently ignored an IOError exception    
    buffer = in_stream.read(CHUNK, exception_on_overflow = False)
    # Convert buffer data from binary to 16 bit int format
    y = np.frombuffer(buffer, dtype = np.int16)
    # Convert to frequency domain
    Y = np.fft.rfft(y)/CHUNK
    # Amplitude of FFT result, the complex ndarray.
    Y_a = np.abs(Y)

    # Band pass filter
    sos = signal.butter(10, [lowcut, highcut], 'bp', fs=SAMPLE_RATE, output='sos')
    filtered = signal.sosfilt(sos, y)
    FILTERED = np.fft.rfft(filtered)/CHUNK
    FILTERED_a = np.abs(FILTERED)
    line0.set_ydata(spectrum_base+FILTERED_a[mask_plot]/divide_factor)
    S_a = FILTERED_a
    # Leave freq. in freq_range. 
    S_a[mask] = 0
    # Find main freq.
    main_freq = freq[np.argmax(S_a)] 

    vline = ax.vlines(d2r(main_freq), 0 , pointer_len, colors=pointer_color,\
                      linewidth= pointer_width, zorder=3)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.0001)
    vline.remove()

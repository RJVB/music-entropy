#!/usr/bin/env python3

"""Command line utility for evaluating song entropy"""

# file I/O
import wave
import struct

# math
import numpy
from numpy.fft import fft, fftfreq, rfft, rfftfreq
import math

# documentation
from api_docs import command, parse_args, help


@command
def get_shannon_rel_entropy(file_name, sample_interval=1, duration=-1):
    """Get ratio of the song's entropy to the entropy of the Uniform
    distribution, optionally over the given duration in seconds.
    <sample_interval> is currently ignored.
    """
    wr = wave.open(file_name, 'r')

    duration=float(duration)
    if duration >= 0:
        print("Reading", duration, "seconds from file, sample interval", sample_interval, "...", end=' ', flush=True)
        song = _time_data(wr,sample_interval=sample_interval, max_frames=int(duration * wr.getframerate()))
    else:
        print("Reading entire file, sample interval", sample_interval, "...", end=' ', flush=True)
        song = _time_data(wr, sample_interval=sample_interval)

    print("\nfft'ing ...", end=' ', flush=True)
    song_fft = fft(song)
    print("length=", len(song_fft))
    at = abs(song_fft).argmax()
    frequencies = fftfreq(len(song), 1.0 / wr.getframerate())
    print("\tPeak(s)", abs(song_fft)[at], "at", frequencies[at], "Hz")
    print("\tDC :", abs(song_fft[0]), "at", frequencies[0], "Hz")

    print("Calculating Shannon entropy and relative entropy ...", end=' ', flush=True)
    entropy = _shannon_rel_entropy(song_fft)
    print(entropy)
    return entropy[1]

@command
def plot(domain, file_name, sample_interval=1, duration=-1):
    """Plot .wav file over time or frequency.
    
    Parameters:
        <domain> can be 'time' or 'freq'. 'freq' plots the magnitudes of
        the Fast Fourier Transform of the .wav data.
    <sample_interval> is currently ignored.

    """
    # try to get my KFusion style
    from os import putenv
    putenv("QT_STYLE_OVERRIDE", "KFusion")

    import matplotlib
    if matplotlib.get_backend() == 'agg':
        # the default Qt backend must not be available; try
        # to fall back on the less-complex-to-install GTk3 backend
        matplotlib.use(backend='GTK3agg')

    if "time" in domain:
        _plot_time(file_name, sample_interval=sample_interval, duration=duration)
    if "freq" in domain:
        _plot_frequencies(file_name, sample_interval=sample_interval, duration=duration)
 
def _plot_time(file_name, sample_interval=1, duration=-1):
    from pylab import plot as pyplot
    from pylab import xlabel, ylabel, title, grid, show

    try:
        sample_interval = int(sample_interval)
    except TypeError:
        print("argument sample_interval must be int")
        raise SystemExit
    wr = wave.open(file_name, 'r')
    frame_rate = wr.getframerate()
    duration=float(duration)
    if duration >= 0:
        maxframes = int(duration * frame_rate)
    else:
        maxframes = None
    song = _time_data(wr, sample_interval=sample_interval, max_frames=maxframes)
    num_frames = int(maxframes) if maxframes else wr.getnframes()

    delta_t = sample_interval / frame_rate
    t = numpy.arange(0.0, (num_frames - sample_interval) / frame_rate + delta_t, delta_t)
    
    pyplot(t, song)

    xlabel('time (s)')
    ylabel('amplitude')
    title('Amplitude of track {} over time'.format(file_name))
    grid(True)
    show(block=True)

def _plot_frequencies(file_name, sample_interval=1, duration=-1):
    from pylab import plot as pyplot
    from pylab import xlabel, ylabel, title, grid, show

    try:
        sample_interval = int(sample_interval)
    except TypeError:
        print("argument sample_interval must be int")
        raise SystemExit

    wr = wave.open(file_name, 'r')
    frame_rate = wr.getframerate()
    duration=float(duration)
    if duration >= 0:
        maxframes = int(duration * frame_rate)
    else:
        maxframes = None
    song = _time_data(wr, sample_interval=sample_interval, max_frames=maxframes)
    frequencies = rfftfreq(len(song), 1 / frame_rate)

    #ax = pyplot.axes(xlabel='frequency (Hz)',
                     #ylabel='amplitude (complex modulus)',
                     #title='Amplitudes of frequencies of track {}'.format(file_name))
    pyplot(frequencies, numpy.abs(rfft(song)))

    xlabel('frequency (Hz)')
    ylabel('amplitude (complex modulus)')
    title('Amplitudes of frequencies of track {}'.format(file_name))
    grid(True)
    show(block=True)

@command
def get_wav_info(file_name):
    """Print meta-info about .wav file.

    Output:
        sample width (in bytes)
        frame rate (in Hz)
        num frames
        track length (in seconds)
        num channels
    """
    wr = wave.open(file_name, 'r')
    sample_width = wr.getsampwidth()
    frame_rate = wr.getframerate()
    num_frames = wr.getnframes()
    n_channels = wr.getnchannels()
    s = "sample width: {} bytes\n".format(sample_width) + \
        "sample class: {}\n".format(wr.getsampclass().__name__) + \
        "frame rate: {} Hz\n".format(frame_rate) + \
        "num frames: {}\n".format(num_frames) + \
        "track length: {} s\n".format(num_frames / frame_rate) + \
        "num channels: {}\n".format(n_channels)
    #if wr.getusedsampwidth() != sample_width:
        #s += "Effective sample width: {} bytes\n".format(wr.getusedsampwidth())
    return s

#import HRTime
def _time_data(wr, sample_interval=1, max_frames=None):
    # expects wav file object opened for reading

    sample_interval=int(sample_interval)
    sample_width = wr.getsampwidth()
    frame_rate = wr.getframerate()
    n_frames = int(max_frames) if max_frames else wr.getnframes()
    n_channels = wr.getnchannels()
    ## if n_channels != 2 or sample_width != 2 or frame_rate != 44100:
    if n_channels != 2 or not sample_width in [2,3,4,8]:
        print("Unsupported sample width or frame rate")
        get_wav_info(wr)
        raise SystemExit

    dim=n_frames

    chunksz = sample_width * 2
    bytesize = n_frames * chunksz
    # let readframes() read in chunks that are approx. 64k bytes
    # worth of frames or the entire file, whichever is smaller.
    # (set blockread to 0 to fall back to the original 1-frame
    # method, preserved for debugging purposes.)
    blockread = int(min(bytesize,64*1024)/chunksz)

    if blockread == 0:
        unpack_format = "<"
    if wr.getsampclass() == int:
        if blockread == 0:
            if sample_width == 8:
                unpack_format += "ll"
                chan = numpy.empty([2,dim], long)
            elif sample_width == 4:
                unpack_format += "ii"
                chan = numpy.empty([2,dim], int)
            elif sample_width == 2:
                unpack_format += "hh"
                chan = numpy.empty([2,dim], numpy.int16)
            else:
                chan = numpy.empty([2,dim], int)
                unpack_format += f'{int(sample_width*2)}B'
        dtype = f'<i{sample_width}'
    elif wr.getsampclass() == float:
        if blockread == 0:
            chan = numpy.empty([2,dim], float)
            unpack_format += f'{int(sample_width/2)}f'
        dtype = f'<f{sample_width}'

    # we assume WAV files with signed samples
    #zero = 0 #(2 ** (sample_width * 8)) / 2

    #HRTime.tic()
    # this probably assumes pcm_sXX samples with the same endianness as the host
    if blockread > 0:
        raw = numpy.empty(dim,dtype=f'S{chunksz}')
        ii = 0
        for i in range(0,dim,blockread):
            wave_data = wr.readframes(blockread)

            # slice up in chunks what we actually read (the last
            # read will most likely not be of size <blockread> -
            # when we're reading the entire file!!)
            splitrange = int(len(wave_data)/chunksz)
            if splitrange + ii >= dim:
                # this can happen when we're reading only part of a file; readframes()
                # does not know about soft EOF so will return a full <blockread> buffer.
                splitrange = dim - ii
            elif splitrange == 0:
                # should never happen, and bumping to 1 will probably raise an error,
                # but that might be better than letting the situation slide?!
                splitrange = 1
            for j in range(splitrange):
                k = j * chunksz
                raw[ii] = wave_data[k:k+chunksz]
                #except:
                    #print(i, j, wave_data, len(wave_data), blockread)
                    #print("raw[", ii, "] = wave_data[", k, ":", k+chunksz, "]=", wave_data[k:k+chunksz])
                    #raise SystemExit
                ii += 1
    # the all-numpy solutions for extracting the channel data from the "raw" array are
    # thanks to 'homer512' (https://stackoverflow.com/a/79847078/1460868)
    if sample_width != 3:
        if blockread == 0:
            for i in range(0,dim):
                wave_data = wr.readframes(1)

                data = struct.unpack(unpack_format, wave_data)
                # this is actually faster than chan[:,i]=data !
                chan[0,i] = data[0]
                chan[1,i] = data[1]
            chan = chan.astype(float)
        else:
            chan = numpy.frombuffer(raw,dtype=dtype).reshape(dim,2).transpose().astype(float)
    else:
        if blockread == 0:
            for i in range(0,dim):
                wave_data = wr.readframes(1)

                data = struct.unpack(unpack_format, wave_data)
                # we assume pcm_s24le!
                chan[0,i] = int.from_bytes(data[0:3], 'little', signed=True)
                chan[1,i] = int.from_bytes(data[3:6], 'little', signed=True)
            chan = chan.astype(float)
        else:
            bytelength = 1 << sample_width.bit_length()
            rightshift = (bytelength - sample_width) * 8 # for sign extension
            udtype = f'u{bytelength}'
            idtype = f'i{bytelength}'
            leftshifts = numpy.arange(rightshift, 8 * bytelength, 8, dtype=udtype)
            chan = ((numpy.frombuffer(raw,dtype=f'<u1').reshape((dim, 2, sample_width)) << leftshifts) \
                    .sum(axis=-1, dtype=udtype).astype(idtype) >> rightshift) \
                    .transpose().astype(float)
    raw = []
    # calculate the average of the 2 channels:
    song = numpy.mean(chan, axis=0)
    max_sample = numpy.max(abs(song))

    print() #"Data read in", HRTime.toc(), "s")

    time_base = float(sample_interval) / float(frame_rate);
    #chan -= zero
    max_amp0 = numpy.max(abs(chan[0]))
    #if max_amp0 == 0:
        #print("chan0 range,av,stdev:", numpy.min(chan0), numpy.max(chan0), numpy.mean(chan0), numpy.std(chan0))
    max_amp1 = numpy.max(abs(chan[1]))
    print("Max abs. channel-0 sample value:", max_amp0)
    print("Max abs. channel-1 sample value:", max_amp1)
    chan[0] /= max_amp0
    chan[1] /= max_amp1
    min_amp0 = numpy.min(chan[0])
    max_amp0 = numpy.max(chan[0])
    print("Normalised chan0 range: [", min_amp0, "at t=", chan[0].argmin() * time_base, "seconds",
          "] -\n\t[", max_amp0, "at t=", chan[0].argmax() * time_base, "seconds ]")
    min_amp1 = numpy.min(chan[1])
    max_amp1 = numpy.max(chan[1])
    print("Normalised chan1 range: [", min_amp1, "at t=", chan[1].argmin() * time_base, "seconds",
          "] -\n\t[", max_amp1, "at t=", chan[1].argmax() * time_base, "seconds ]")

    #song -= zero;
    print("\nMax abs. channel-mix sample value:", max_sample)
    song /= max_sample
    min_sample = numpy.min(song)
    max_sample = numpy.max(song)
    print("Normalised range: [", min_sample, "at t=", song.argmin() * time_base, "seconds",
          "] -\n\t[", max_sample, "at t=", song.argmax() * time_base, "seconds ]")

    return song

def _shannon_rel_entropy(song_fft):
    entropy = _entropy(song_fft)
    return numpy.array([entropy, entropy / math.log(len(song_fft), 2)])

#import HRTime
def _entropy(song_fft):
    # normalise the fft
    #HRTime.tic()

    #total_weight = sum([abs(z) for z in song_fft])
    #total_weight = sum(numpy.abs(song_fft))
    #song_entropy = 0
    #p_x_sum = 0
    #for z in song_fft:
        #p_x = (abs(z)/total_weight)
        #p_x_sum += p_x
        #song_entropy += p_x * math.log(1/p_x, 2)

    # use numpy's array arithmatic, which is a good 5x faster
    total_weight = sum(numpy.abs(song_fft))
    p_x = numpy.abs(song_fft) / total_weight
    song_entropy = sum( p_x * numpy.log2(1/p_x) )

    #print("entropy calculated in", HRTime.toc(), "s")
    return song_entropy

if __name__ == '__main__':
    parse_args()

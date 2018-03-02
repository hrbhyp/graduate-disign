function [traindata] = passband_filter(signal)

Y = fft(signal,size(signal,1));
Pyy = Y.*conj(Y)/size(signal,1);
f = 240/size(signal,1)*(0:(size(signal,1)/2-1));

d = fdesign.bandpass('N,Fst1,Fp1,Fp2,Fst2,C',40,0.08,0.1,20,24,240);
d.Stopband1Constrained = true;
d.Astop1 = 60;
d.Stopband2Constrained = true;
d.Astop2 = 60;

Hd = design(d,'equiripple');

traindata = filter(Hd,signal);
for i=1:size(signal,1)
    for j=1:size(signal,2)
        if(i<=lowpass || i>=highpass)
            signalfft(i,j)=0;
        end
    end
end
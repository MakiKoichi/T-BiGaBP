function [H] = MYGaussianChannelMatrix(Nr,Nt,duration,time,P)
% MYGaussianChannelMatrix : Generate a matrix with each element following an independent complex Gaussian distribution varying with a constant period
% 
% Parameters
% ----------
% Nr : Number of rows in output matrix
% 
% Nr : Number of output matrix columns
% 
% duration : Matrix duration
%
% time : Total discrete time
% 
% Returns
% ----------
% H : 3D array of [Nr Nt time] ([Nr Nt] matrix are stored in time in 3D direction )

prt = fix(time / duration);
H = zeros(Nr,Nt,time);
for i = 0:prt-1
    matrix = MYcompNoise([Nr,Nt],P);
    noise = zeros(Nr,Nt,duration);
    for j=1:duration
    noise(:,:,j) = matrix;
    end
    H(:,:,i*duration+1:(i+1)*duration) = noise;
end
if time > duration*prt
    matrix = MYcompNoise([Nr,Nt],P);
    noise = zeros(Nr,Nt,time-duration*prt);
    for j=1:time-duration*prt
    noise(:,:,j) = matrix;
    H(:,:,duration*prt+1:time) = noise;
    end
end
end

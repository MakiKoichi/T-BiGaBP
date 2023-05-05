function[noise] = MYcompNoise(noiseSize, Pn)
% MYcompNoise : Generate complex Gaussian noise matrix with mean zero
% 
% Parameters
% ----------
% noiseSize : Size of the output noise matrix
% 
% Pn : Noise power
%
% Returns
% ----------
% noise : Noise matrix

noise = ( randn(noiseSize) + j*randn(noiseSize) ) * sqrt(Pn/2);
end

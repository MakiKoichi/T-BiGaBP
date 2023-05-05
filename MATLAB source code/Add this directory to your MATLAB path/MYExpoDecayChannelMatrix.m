function [H] = MYExpoDecayChannelMatrix(Theta,M,N,Phi)
% MYExpoDecayChannelMatrix : Generate channel matrix in the presence of RX spatial correlation
% 
% [H] = MYExpoDecayChannelMatrix(rho,M,N)
% 
% Parameters
% ----------
% Theta : RX spatial matrix
% 
% M : Number of UE
%
% N : Number of RX antennas of BS
% 
% Returns
% ----------
% H : [N M]Channel matrix

H = sqrtm(Theta)*MYGaussianChannelMatrix(N,M,1,1,Phi);
end

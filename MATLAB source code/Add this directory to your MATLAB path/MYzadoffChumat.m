function [Z] = MYzadoffChumat(R,M,Kp,l)
% MYzadoffChumat Zadoff-Chu系列に基づいたパイロット行列を生成
%
% Parameters
% ----------
% R : Root of Zadoff-Chu sequence
%
% M : Number of UE
% 
% Kp : Length of pilot symbols
%
% l : Integer for Zadoff-Chu sequence
%
% Returns
% ----------
% Z : [Kp Kp] Pilot matrix

if Kp >= M
    N = Kp;
else
    N = M;
end

if rem(N,2) == 0
    z = exp(-j*2*pi*R/N*((0:N-1)'.^2/2+l*(0:N-1)'));
else
    z = exp(-j*2*pi*R/N*((0:N-1)'.*(1:N)'/2+l*(0:N-1)'));
end
    
Z_all = toeplitz([z(1) fliplr(z(2:end).')], z.');

if Kp >= M
    Z = Z_all(1:M,:);
else
    Z = Z_all(:,1:Kp);
end

end

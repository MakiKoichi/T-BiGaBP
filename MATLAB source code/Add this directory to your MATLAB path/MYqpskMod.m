function [QPSKsymbol] = MYqpskMod(data)
% MYqpskMod : QPSK modulator
%
% Parameters
% ----------
% data : Binary data (column vector)
%
% Returns
% ----------
% QPSKsymbol : QPSK symbols (column vector)

Ndata = length(data);                   
spcOutput = reshape(data,2,Ndata/2);    %Series-parallel conversion
QPSKsymbolIndex = [1,2]*spcOutput;      %Conversion to decimal
QPSKsymbol = ones(1,Ndata/2)*exp(j*pi/4);
QPSKsymbol(find(QPSKsymbolIndex==1)) = exp(j*3*pi/4);
QPSKsymbol(find(QPSKsymbolIndex==3)) = exp(j*5*pi/4);
QPSKsymbol(find(QPSKsymbolIndex==2)) = exp(j*7*pi/4);
QPSKsymbol=QPSKsymbol(:);
end

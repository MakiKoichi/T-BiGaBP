function[data] = MYrandData(Ndata)
% MYrandData : Random binary data generator
%
% Parameters
% ----------
% Ndata : Number of data bits
%
% Returns
% ----------
% data : Random binary data vector

dataSeq = randn(Ndata,1);
data = zeros(Ndata,1);
data(find(dataSeq>0)) = 1;
return

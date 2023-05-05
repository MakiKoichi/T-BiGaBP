%% System model

M  = 16;                                 % Number of UE
N  = 32;                                 % Number of RX antennas of BS
Kp = 16;                                 % Length of pilot symbols
Kd = 256;                                % Length of data symbols
K  = Kp + Kd;                            % Total symbol frame
Phi = 1/N;                               % fading factor
Es = 1;                                  % average power density of constellations
cx = sqrt(Es/2);                         
R = 25;                                  % Root of Zadoff-Chu sequence
l = -1;                                  % Integer for Zadoff-Chu sequence
Xp = sqrt(Es)*MYzadoffChumat(R,M,Kp,l);  % Pilot matrix
rho = 0.7;                               % Correlation coefficient
SNRdB = 23;                              % SNR [dB]
N0    = 10^(-SNRdB/10);                  % Noise spectral density
T = 32;                                  % Number of iterations

%% Simulation setup

a = 4;                       % Gain of sigmoid function
MINI_BATCH = 3000;           % Number of mini batches
BATCH_SIZE = 100;            % Batch size
EPOCH = 1;                   % Number of epochs
STEP_SIZE  = 5e-3;           % Step size of SGD

%% Loading parameters in progress

fileList = dir('*.mat');
max_epoch = 0;
for n = 1:length(fileList)
    fileName = fileList(n).name;
    if contains(fileName,'ExpoDecay')
        newStr = extract(fileName,digitsPattern);
        if str2num(cell2mat(newStr(1))) > max_epoch
            max_epoch = str2num(cell2mat(newStr(1)));
            batchsize = str2num(cell2mat(newStr(3)));
        end
    end
end

if max_epoch ~= 0
    fileList = dir(['ExpoDecay_epoch_',num2str(max_epoch),'*.mat']);
    max_minibatch = 1;
    for n = 1:length(fileList)
        fileName = fileList(n).name;
        newStr = extract(fileName,digitsPattern);
        if str2num(cell2mat(newStr(2))) > max_minibatch;
            max_minibatch = str2num(cell2mat(newStr(2)));
        end
    end
end

if max_epoch ~= 0
    fileName = ['ExpoDecay_epoch_',num2str(max_epoch),'_minibatch_',num2str(max_minibatch),'_batchsize_',num2str(batchsize),'.mat'];
    if max_minibatch == BATCH_SIZE
        init_epoch = max_epoch + 1;
        init_minibatch = 1;
    else
        init_epoch = max_epoch;
        init_minibatch = max_minibatch + 1;
    end
    load(fileName)
else
    zeta_x = 0.5*ones(N,T-1);
    zeta_h = 0.5*ones(N,T-1);
    eta_x = zeros(N,T-1);
    eta_h = zeros(N,T-1);
    alpha = ones(1,T-1);
    beta  = ones(1,T-1);
    gamma = ones(1,T-1);
    init_epoch = 1;
    init_minibatch = 1;
end

clear fileList n fileName newStr batchsize max_epoch max_minibatch

if exist('Losslist.mat')
    load('Losslist.mat','Losslist')
else
    Losslist = [];
end

%% Optimization

for epoch = init_epoch:EPOCH
for minibatch = init_minibatch:MINI_BATCH

    eta_x_backward   = zeros(N,T-1);
    eta_h_backward   = zeros(N,T-1);
    alpha_backward   = zeros(1,T-1);
    beta_backward    = zeros(1,T-1);
    gamma_backward   = zeros(1,T-1);
    
    Loss = 0;
    
    for batch_size = 1:BATCH_SIZE
        
        d = MYrandData(2*M*Kd);
        d_modulated = MYqpskMod(d);
        Xd = (reshape(d_modulated,[Kd,M])).';
        X = [Xp Xd];
        Theta = rho.^(abs(repmat((1:N).',1,N)-repmat(1:N,N,1)));
        H = MYExpoDecayChannelMatrix(Theta,M,N,Phi);
        Y = H*X + MYcompNoise([N,K], N0);   
        
        % Initial channel estimation
        
        %%%  Using orthogonal pilots : least square %%%
        
        if Kp >= M
            H_hat     = Y(:,1:Kp) * X(:,1:Kp)' * ( X(:,1:Kp)*X(:,1:Kp)' )^(-1);
            psi_h_hat = Es/Kp*N0;
            
        %%% Using non-orthogonal pilots : minimum norm solution %%%
        
        else
            H_hat = Y(:,1:Kp) * (X(:,1:Kp)'*X(:,1:Kp))^(-1)*X(:,1:Kp)';
            Gp = X(:,1:Kp)*(X(:,1:Kp)'*X(:,1:Kp))^(-1)*X(:,1:Kp)';
            psi_h_hat = Phi/M * trace( (Gp-eye(M))'*(Gp-eye(M)) ) + N0/M * trace( (X(:,1:Kp)'*X(:,1:Kp))^(-1) );
        end

                x_hat = zeros(M,K,N,T-1);
        psi_x = zeros(M,K,N,T-2);
        x_var = zeros(M,Kd,N,T-1);
        psi_r = zeros(M,Kd,N,T-1);
        r_hat = zeros(M,Kd,N,T-1);

        y_tilde = zeros(N,K,M,T);
        nu_y = zeros(N,K,M,T);
        nu_x = zeros(N,Kd,M,T);
        nu_h = zeros(N,K,M,T-1);                

        x_hat(:,1:Kp,:,:)   = repmat(Xp,[1,1,N,T-1]);
        psi_x(:,Kp+1:K,:,:) = Es;
        h_hat = zeros(N,M,K,T-1);
        h_hat(:,:,:,1) = repmat(H_hat,[1,1,K]);
        psi_h = zeros(N,M,K,T-1);
        psi_h(:,:,:,1) = repmat(psi_h_hat,[N,M,K]);
        psi_q = zeros(N,M,K,T-1);
        q_hat = zeros(N,M,K,T-1);

        re_x_var_backward = zeros(M,Kd,N,T);
        im_x_var_backward = zeros(M,Kd,N,T);
        re_x_hat_backward = zeros(M,K,N,T);
        im_x_hat_backward = zeros(M,K,N,T);
        psi_x_backward = zeros(M,K,N,T);
        re_q_hat_backward = zeros(N,M,K,T-1);
        im_q_hat_backward = zeros(N,M,K,T-1);
        re_h_hat_backward = zeros(N,M,K,T);
        im_h_hat_backward = zeros(N,M,K,T);
        psi_h_backward    = zeros(N,M,K,T);
        re_r_hat_backward = zeros(M,Kd,N,T-1);
        im_r_hat_backward = zeros(M,Kd,N,T-1);
        psi_r_backward    = zeros(M,Kd,N,T-1);
        psi_q_backward    = zeros(N,M,K,T-1);
        nu_h_backward     = zeros(N,K,M,T-1);
        re_y_tilde_backward = zeros(N,K,M,T);
        im_y_tilde_backward = zeros(N,K,M,T);
        nu_y_backward     = zeros(N,K,M,T);
        nu_x_backward     = zeros(N,Kd,M,T);
        zeta_x_backward   = zeros(N,T-1);
        zeta_h_backward   = zeros(N,T-1);

       %% forward

        for t = 1:T
            
            temp = permute(h_hat(:,:,:,t),[1,3,2]).*permute(x_hat(:,:,:,t),[3,2,1]);
            y_tilde(:,:,:,t) = Y - (repmat(sum(temp,3),1,1,M)-temp);
            
            temp = abs(permute(h_hat(:,:,:,t),[1,3,2])).^2.*permute(psi_x(:,:,:,t),[3,2,1])+(abs(permute(x_hat(:,:,:,t),[3,2,1])).^2+permute(psi_x(:,:,:,t),[3,2,1])).*permute(psi_h(:,:,:,t),[1,3,2]);
            nu_y(:,:,:,t) = sum(temp,3)- temp + N0;

            nu_x(:,:,:,t) = nu_y(:,Kp+1:K,:,t) + Es*permute(psi_h(:,:,Kp+1:K,t),[1,3,2]);   
            
            if t ~= T
            
            temp = abs(permute(h_hat(:,:,Kp+1:K,t),[2,3,1])).^2./permute(nu_x(:,:,:,t),[3,2,1]);
            psi_r(:,:,:,t) = (repmat(sum(temp,3),1,1,N) - temp).^(-1);           

            temp = conj(permute(h_hat(:,:,Kp+1:K,t),[2,3,1])).*permute(y_tilde(:,Kp+1:K,:,t),[3,2,1])./permute(nu_x(:,:,:,t),[3,2,1]);
            r_hat(:,:,:,t) = psi_r(:,:,:,t).*(repmat(sum(temp,3),1,1,N)-temp);

            nu_h(:,:,:,t) = nu_y(:,:,:,t) + Phi*permute(psi_x(:,:,:,t),[3,2,1]);
            
            temp = cat(3,alpha(t)*Es./permute(nu_h(:,1:Kp,:,t),[1,3,2]),beta(t)*abs(permute(x_hat(:,Kp+1:K,:,t),[3,1,2])).^2./permute(nu_h(:,Kp+1:K,:,t),[1,3,2]));
            psi_q(:,:,:,t) = (repmat(sum(temp,3),1,1,K)-temp).^(-1);
            
            temp = cat(3,alpha(t)*conj(permute(x_hat(:,1:Kp,:,t),[3,1,2])).*permute(y_tilde(:,1:Kp,:,t),[1,3,2])./permute(nu_h(:,1:Kp,:,t),[1,3,2]),beta(t)*conj(permute(x_hat(:,Kp+1:K,:,t),[3,1,2])).*permute(y_tilde(:,Kp+1:K,:,t),[1,3,2])./permute(nu_h(:,Kp+1:K,:,t),[1,3,2]));
            q_hat(:,:,:,t) = psi_q(:,:,:,t).*(repmat(sum(temp,3),1,1,K)-temp);

            x_var(:,:,:,t+1) = cx.*(tanh(gamma(t)/cx.*real(r_hat(:,:,:,t)))+j*tanh(gamma(t)/cx.*imag(r_hat(:,:,:,t))));

            x_hat(:,Kp+1:K,:,t+1) = permute(repmat(zeta_x(:,t),1,Kd,M),[3,2,1]).*x_var(:,:,:,t+1) + (1-permute(repmat(zeta_x(:,t),1,Kd,M),[3,2,1])).*x_hat(:,Kp+1:K,:,t);

            psi_x(:,Kp+1:K,:,t+1) = permute(repmat(zeta_x(:,t),1,Kd,M),[3,2,1]).*(Es-abs(x_var(:,:,:,t+1)).^2) + (1-permute(repmat(zeta_x(:,t),1,Kd,M),[3,2,1])).*psi_x(:,Kp+1:K,:,t);

            h_hat(:,:,:,t+1) = permute(repmat(zeta_h(:,t),1,K,M),[1,3,2]).*Phi.*q_hat(:,:,:,t)./(psi_q(:,:,:,t)+Phi) + (1-permute(repmat(zeta_h(:,t),1,K,M),[1,3,2])).*h_hat(:,:,:,t);

            psi_h(:,:,:,t+1) = permute(repmat(zeta_h(:,t),1,K,M),[1,3,2]).*Phi.*psi_q(:,:,:,t)./(psi_q(:,:,:,t)+Phi) + (1-permute(repmat(zeta_h(:,t),1,K,M),[1,3,2])).*psi_h(:,:,:,t);    
            
            end
            
        end
        
        clear temp
        
        psi_r_joint = sum(abs(permute(h_hat(:,:,Kp+1:K,T),[2,3,1])).^2./permute(nu_x(:,:,:,T),[3,2,1]),3).^(-1);

        R_hat = psi_r_joint.*sum(conj(permute(h_hat(:,:,Kp+1:K,T),[2,3,1])).*permute(y_tilde(:,Kp+1:K,:,T),[3,2,1])./permute(nu_x(:,:,:,T),[3,2,1]),3);

       %% backward
        
        Loss = Loss + sum(sum(abs(Xd-R_hat).^2,2))/(M*Kd);
       
        re_R_hat_backward = -2*(real(Xd)-real(R_hat))/(M*Kd);
        im_R_hat_backward = -2*(imag(Xd)-imag(R_hat))/(M*Kd);

        re_y_tilde_backward(:,Kp+1:K,:,T) = permute(repmat(psi_r_joint,1,1,N),[3,2,1]).*real(permute(h_hat(:,:,Kp+1:K,T),[1,3,2]))./nu_x(:,:,:,T).*permute(repmat(re_R_hat_backward,1,1,N),[3,2,1])...
                                           -permute(repmat(psi_r_joint,1,1,N),[3,2,1]).*imag(permute(h_hat(:,:,Kp+1:K,T),[1,3,2]))./nu_x(:,:,:,T).*permute(repmat(im_R_hat_backward,1,1,N),[3,2,1]);
        im_y_tilde_backward(:,Kp+1:K,:,T) = permute(repmat(psi_r_joint,1,1,N),[3,2,1]).*imag(permute(h_hat(:,:,Kp+1:K,T),[1,3,2]))./nu_x(:,:,:,T).*permute(repmat(re_R_hat_backward,1,1,N),[3,2,1])...
                                           +permute(repmat(psi_r_joint,1,1,N),[3,2,1]).*real(permute(h_hat(:,:,Kp+1:K,T),[1,3,2]))./nu_x(:,:,:,T).*permute(repmat(im_R_hat_backward,1,1,N),[3,2,1]);
                                       
        psi_r_joint_backward = real(R_hat)./psi_r_joint.*re_R_hat_backward + imag(R_hat)./psi_r_joint.*im_R_hat_backward;
        
        nu_x_backward(:,:,:,T) = -permute(repmat(psi_r_joint,1,1,N),[3,2,1]).*(real(permute(h_hat(:,:,Kp+1:K,T),[1,3,2])).*real(y_tilde(:,Kp+1:K,:,T))+imag(permute(h_hat(:,:,Kp+1:K,T),[1,3,2])).*imag(y_tilde(:,Kp+1:K,:,T)))./(nu_x(:,:,:,T)).^2.*permute(repmat(re_R_hat_backward,1,1,N),[3,2,1])...
                                 -permute(repmat(psi_r_joint,1,1,N),[3,2,1]).*(real(permute(h_hat(:,:,Kp+1:K,T),[1,3,2])).*imag(y_tilde(:,Kp+1:K,:,T))-imag(permute(h_hat(:,:,Kp+1:K,T),[1,3,2])).*real(y_tilde(:,Kp+1:K,:,T)))./(nu_x(:,:,:,T)).^2.*permute(repmat(im_R_hat_backward,1,1,N),[3,2,1])...
                                 +permute(abs(h_hat(:,:,Kp+1:K,T)).^2,[1,3,2])./(nu_x(:,:,:,T).^2).*permute(repmat(psi_r_joint,1,1,N),[3,2,1]).^2.*permute(repmat(psi_r_joint_backward,1,1,N),[3,2,1]);
                             
        nu_y_backward(:,Kp+1:K,:,T) = nu_x_backward(:,:,:,T);
        
        %%%
        partsum_re_y_tilde_backward = repmat(sum(re_y_tilde_backward(:,:,:,T),3),1,1,M)-re_y_tilde_backward(:,:,:,T);
        partsum_im_y_tilde_backward = repmat(sum(im_y_tilde_backward(:,:,:,T),3),1,1,M)-im_y_tilde_backward(:,:,:,T);
        partsum_nu_y_backward       = repmat(sum(      nu_y_backward(:,:,:,T),3),1,1,M)-      nu_y_backward(:,:,:,T);
        %%%
        
        re_x_hat_backward(:,:,:,T) = -real(permute(h_hat(:,:,:,T),[2,3,1])).*permute(partsum_re_y_tilde_backward,[3,2,1])...
                                     -imag(permute(h_hat(:,:,:,T),[2,3,1])).*permute(partsum_im_y_tilde_backward,[3,2,1])...
                                     +2*real(x_hat(:,:,:,T)).*permute(psi_h(:,:,:,T),[2,3,1]).*permute(partsum_nu_y_backward,[3,2,1]);
        im_x_hat_backward(:,:,:,T) =  imag(permute(h_hat(:,:,:,T),[2,3,1])).*permute(partsum_re_y_tilde_backward,[3,2,1])...
                                     -real(permute(h_hat(:,:,:,T),[2,3,1])).*permute(partsum_im_y_tilde_backward,[3,2,1])...
                                     +2*imag(x_hat(:,:,:,T)).*permute(psi_h(:,:,:,T),[2,3,1]).*permute(partsum_nu_y_backward,[3,2,1]);
                                 
        psi_x_backward(:,:,:,T) = (abs(permute(h_hat(:,:,:,T),[2,3,1])).^2 + permute(psi_h(:,:,:,T),[2,3,1])).*permute(partsum_nu_y_backward,[3,2,1]);
        
        psi_h_backward(:,:,:,T) =  (abs(permute(x_hat(:,:,:,T),[3,1,2])).^2+permute(psi_x(:,:,:,T),[3,1,2])).*permute(partsum_nu_y_backward,[1,3,2])...
                                      +cat(3,zeros(N,M,Kp),Es*permute(nu_x_backward(:,:,:,T),[1,3,2]));
        
        re_h_hat_backward(:,:,:,T) = -real(permute(x_hat(:,:,:,T),[3,1,2])).*permute(partsum_re_y_tilde_backward,[1,3,2])...
                                     -imag(permute(x_hat(:,:,:,T),[3,1,2])).*permute(partsum_im_y_tilde_backward,[1,3,2])...
                                     +2*real(h_hat(:,:,:,T)).*permute(psi_x(:,:,:,T),[3,1,2]).*permute(partsum_nu_y_backward,[1,3,2])...
                                     +cat(3,zeros(N,M,Kp),permute(repmat(psi_r_joint,1,1,N),[3,1,2]).*real(permute(y_tilde(:,Kp+1:K,:,T),[1,3,2]))./permute(nu_x(:,:,:,T),[1,3,2]).*permute(repmat(re_R_hat_backward,1,1,N),[3,1,2])...
                                                         +permute(repmat(psi_r_joint,1,1,N),[3,1,2]).*imag(permute(y_tilde(:,Kp+1:K,:,T),[1,3,2]))./permute(nu_x(:,:,:,T),[1,3,2]).*permute(repmat(im_R_hat_backward,1,1,N),[3,1,2])...
                                                         -2*real(h_hat(:,:,Kp+1:K,T))./permute(nu_x(:,:,:,T),[1,3,2]).*permute(repmat(psi_r_joint,1,1,N),[3,1,2]).^2.*permute(repmat(psi_r_joint_backward,1,1,N),[3,1,2]));
        im_h_hat_backward(:,:,:,T) =  imag(permute(x_hat(:,:,:,T),[3,1,2])).*permute(partsum_re_y_tilde_backward,[1,3,2])...
                                     -real(permute(x_hat(:,:,:,T),[3,1,2])).*permute(partsum_im_y_tilde_backward,[1,3,2])...
                                     +2*imag(h_hat(:,:,:,T)).*permute(psi_x(:,:,:,T),[3,1,2]).*permute(partsum_nu_y_backward,[1,3,2])...
                                     +cat(3,zeros(N,M,Kp),permute(repmat(psi_r_joint,1,1,N),[3,1,2]).*imag(permute(y_tilde(:,Kp+1:K,:,T),[1,3,2]))./permute(nu_x(:,:,:,T),[1,3,2]).*permute(repmat(re_R_hat_backward,1,1,N),[3,1,2])...
                                                         -permute(repmat(psi_r_joint,1,1,N),[3,1,2]).*real(permute(y_tilde(:,Kp+1:K,:,T),[1,3,2]))./permute(nu_x(:,:,:,T),[1,3,2]).*permute(repmat(im_R_hat_backward,1,1,N),[3,1,2])...
                                                         -2*imag(h_hat(:,:,Kp+1:K,T))./permute(nu_x(:,:,:,T),[1,3,2]).*permute(repmat(psi_r_joint,1,1,N),[3,1,2]).^2.*permute(repmat(psi_r_joint_backward,1,1,N),[3,1,2]));
                                                  
        clear partsum_re_y_tilde_backward partsum_im_y_tilde_backward partsum_nu_y_backward

        for t = T-1:-1:1
            
            zeta_x_backward(:,t) = sum(sum(permute( (real(x_var(:,:,:,t+1))-real(x_hat(:,Kp+1:K,:,t))).*re_x_hat_backward(:,Kp+1:K,:,t+1)...
                                                   +(imag(x_var(:,:,:,t+1))-imag(x_hat(:,Kp+1:K,:,t))).*im_x_hat_backward(:,Kp+1:K,:,t+1)...
                                                   +(Es-abs(x_hat(:,Kp+1:K,:,t+1)).^2-psi_x(:,Kp+1:K,:,t)) .*psi_x_backward(:,Kp+1:K,:,t+1),[3,2,1]),3),2);
                                               
            zeta_h_backward(:,t) = sum(sum(permute( ((Phi*real(q_hat(:,:,:,t)))./(psi_q(:,:,:,t)+Phi)-real(h_hat(:,:,:,t))).*re_h_hat_backward(:,:,:,t+1)...
                                                   +((Phi*imag(q_hat(:,:,:,t)))./(psi_q(:,:,:,t)+Phi)-imag(h_hat(:,:,:,t))).*im_h_hat_backward(:,:,:,t+1)...
                                                   +((Phi*psi_q(:,:,:,t))./(psi_q(:,:,:,t)+Phi)-psi_h(:,:,:,t)).*psi_h_backward(:,:,:,t+1),[1,3,2]),3),2);
            
            eta_x_backward(:,t) = eta_x_backward(:,t) + a*exp(-a*eta_x(:,t))./(1+exp(-a*eta_x(:,t))).^2.*zeta_x_backward(:,t);
            eta_h_backward(:,t) = eta_h_backward(:,t) + a*exp(-a*eta_h(:,t))./(1+exp(-a*eta_h(:,t))).^2.*zeta_h_backward(:,t);
            
            re_x_var_backward(:,:,:,t+1) = permute(repmat(zeta_x(:,t),1,Kd,M),[3,2,1]).*re_x_hat_backward(:,Kp+1:K,:,t+1)-2*permute(repmat(zeta_x(:,t),1,Kd,M),[3,2,1]).*real(x_var(:,:,:,t+1)).*psi_x_backward(:,Kp+1:K,:,t+1);
            im_x_var_backward(:,:,:,t+1) = permute(repmat(zeta_x(:,t),1,Kd,M),[3,2,1]).*im_x_hat_backward(:,Kp+1:K,:,t+1)-2*permute(repmat(zeta_x(:,t),1,Kd,M),[3,2,1]).*imag(x_var(:,:,:,t+1)).*psi_x_backward(:,Kp+1:K,:,t+1);
            
            gamma_backward(t) = gamma_backward(t) + sum(sum(sum(  real(r_hat(:,:,:,t)).*sech(real(r_hat(:,:,:,t)/cx*gamma(t))).^2.*re_x_var_backward(:,:,:,t+1)...
                                                                 +imag(r_hat(:,:,:,t)).*sech(imag(r_hat(:,:,:,t)/cx*gamma(t))).^2.*im_x_var_backward(:,:,:,t+1)),2),3);
                                                             
            re_q_hat_backward(:,:,:,t  ) = repmat(zeta_h(:,t),1,M,K).*(Phi./(psi_q(:,:,:,t)+Phi)).*re_h_hat_backward(:,:,:,t+1);
            im_q_hat_backward(:,:,:,t  ) = repmat(zeta_h(:,t),1,M,K).*(Phi./(psi_q(:,:,:,t)+Phi)).*im_h_hat_backward(:,:,:,t+1);
            
            psi_q_backward(:,:,:,t)      = (real(q_hat(:,:,:,t)).*re_q_hat_backward(:,:,:,t) + imag(q_hat(:,:,:,t)).*im_q_hat_backward(:,:,:,t))./psi_q(:,:,:,t)...
                                           + repmat(zeta_h(:,t),1,M,K)./(psi_q(:,:,:,t)+Phi).^2*Phi.*( - real(q_hat(:,:,:,t)) .* re_h_hat_backward(:,:,:,t+1)...
                                                                                                       - imag(q_hat(:,:,:,t)) .* im_h_hat_backward(:,:,:,t+1)...
                                                                                                       + Phi .* psi_h_backward(:,:,:,t+1));
                                                                                                   
            temp1 = (real(permute(x_hat(:,:,:,t),[3,1,2])).*real(permute(y_tilde(:,:,:,t),[1,3,2]))+imag(permute(x_hat(:,:,:,t),[3,1,2])).*imag(permute(y_tilde(:,:,:,t),[1,3,2])))./permute(nu_h(:,:,:,t),[1,3,2]);
            temp2 = repmat(sum(temp1(:,:,1:Kp),3),1,1,K) - cat(3,temp1(:,:,1:Kp),zeros(N,M,Kd));
            temp3 = (real(permute(x_hat(:,:,:,t),[3,1,2])).*imag(permute(y_tilde(:,:,:,t),[1,3,2]))-imag(permute(x_hat(:,:,:,t),[3,1,2])).*real(permute(y_tilde(:,:,:,t),[1,3,2])))./permute(nu_h(:,:,:,t),[1,3,2]);
            temp4 = repmat(sum(temp3(:,:,1:Kp),3),1,1,K) - cat(3,temp3(:,:,1:Kp),zeros(N,M,Kd));
            temp5 = abs(permute(x_hat(:,:,:,t),[3,1,2])).^2./permute(nu_h(:,:,:,t),[1,3,2]);
            temp6 = repmat(sum(temp5(:,:,1:Kp),3),1,1,K) - cat(3,temp5(:,:,1:Kp),zeros(N,M,Kd));
            
            alpha_backward(t) = alpha_backward(t) + sum(sum(sum( psi_q(:,:,:,t).*(temp2.*re_q_hat_backward(:,:,:,t)+temp4.*im_q_hat_backward(:,:,:,t))-psi_q(:,:,:,t).^2.*temp6.*psi_q_backward(:,:,:,t) ),2),3);
            
            clear temp2 temp4 temp6
            
            temp2 = repmat(sum(temp1(:,:,Kp+1:K),3),1,1,K) - cat(3,zeros(N,M,Kp),temp1(:,:,Kp+1:K));
            temp4 = repmat(sum(temp3(:,:,Kp+1:K),3),1,1,K) - cat(3,zeros(N,M,Kp),temp3(:,:,Kp+1:K));
            temp6 = repmat(sum(temp5(:,:,Kp+1:K),3),1,1,K) - cat(3,zeros(N,M,Kp),temp5(:,:,Kp+1:K));
            
            beta_backward(t)  = beta_backward(t)  + sum(sum(sum( psi_q(:,:,:,t).*(temp2.*re_q_hat_backward(:,:,:,t)+temp4.*im_q_hat_backward(:,:,:,t))-psi_q(:,:,:,t).^2.*temp6.*psi_q_backward(:,:,:,t) ),2),3);
            
            clear temp1 temp2 temp3 temp4 temp5 temp6
                                                                                                   
            if t ~= 1

            re_r_hat_backward(:,:,:,t  ) = sech(real(r_hat(:,:,:,t)/cx*gamma(t))).^2*gamma(t).*re_x_var_backward(:,:,:,t+1);
            im_r_hat_backward(:,:,:,t  ) = sech(imag(r_hat(:,:,:,t)/cx*gamma(t))).^2*gamma(t).*im_x_var_backward(:,:,:,t+1);
            
            %%%%%%%%%%                                                                                       
            partsum_psi_q_times_re_q_hat_backward     = repmat(sum(psi_q(:,:,:,t).*re_q_hat_backward(:,:,:,t),3),1,1,K) - psi_q(:,:,:,t).*re_q_hat_backward(:,:,:,t);
            partsum_psi_q_times_im_q_hat_backward     = repmat(sum(psi_q(:,:,:,t).*im_q_hat_backward(:,:,:,t),3),1,1,K) - psi_q(:,:,:,t).*im_q_hat_backward(:,:,:,t);
            partsum_psi_q_square_times_psi_q_backward = repmat(sum(psi_q(:,:,:,t).^2.*psi_q_backward(:,:,:,t),3),1,1,K) - psi_q(:,:,:,t).^2.*psi_q_backward(:,:,:,t);
            %%%%%%%%%%
            
            nu_h_backward(:,:,:,t) = cat(2,alpha(t)*ones(N,Kp,M),beta(t)*ones(N,Kd,M)).*(-(real(permute(x_hat(:,:,:,t),[3,2,1])).*real(y_tilde(:,:,:,t))+imag(permute(x_hat(:,:,:,t),[3,2,1])).*imag(y_tilde(:,:,:,t)))...
                                                                                          .*permute(partsum_psi_q_times_re_q_hat_backward,[1,3,2])...
                                                                                          -(real(permute(x_hat(:,:,:,t),[3,2,1])).*imag(y_tilde(:,:,:,t))-imag(permute(x_hat(:,:,:,t),[3,2,1])).*real(y_tilde(:,:,:,t)))...
                                                                                          .*permute(partsum_psi_q_times_im_q_hat_backward,[1,3,2])...
                                                                                          +abs(permute(x_hat(:,:,:,t),[3,2,1])).^2.*permute(partsum_psi_q_square_times_psi_q_backward,[1,3,2]))./nu_h(:,:,:,t).^2;
        

            psi_r_backward(:,:,:,t) = (real(r_hat(:,:,:,t)).*re_r_hat_backward(:,:,:,t)+imag(r_hat(:,:,:,t).*im_r_hat_backward(:,:,:,t)))./psi_r(:,:,:,t);
            
            
            %%%%%%%%%%
            temp = psi_r(:,:,:,t).*re_r_hat_backward(:,:,:,t);
            partsum_psi_r_times_re_r_hat_backward = repmat(sum(temp,3),1,1,N) - temp;
            temp = psi_r(:,:,:,t).*im_r_hat_backward(:,:,:,t);
            partsum_psi_r_times_im_r_hat_backward = repmat(sum(temp,3),1,1,N) - temp;
            temp = psi_r(:,:,:,t).^2.*psi_r_backward(:,:,:,t);
            partsum_psi_r_square_times_psi_r_backward = repmat(sum(temp,3),1,1,N) - temp;
            clear temp
            %%%%%%%%%%
            
            re_y_tilde_backward(:,:,:,t) = cat(2,alpha(t)*ones(N,Kp,M),beta(t)*ones(N,Kd,M)).*(real(permute(x_hat(:,:,:,t),[3,2,1])).*permute(partsum_psi_q_times_re_q_hat_backward,[1,3,2])-imag(permute(x_hat(:,:,:,t),[3,2,1])).*permute(partsum_psi_q_times_im_q_hat_backward,[1,3,2]))./nu_h(:,:,:,t)...
                                           +cat(2,zeros(N,Kp,M),(real(permute(h_hat(:,:,Kp+1:K,t),[1,3,2])).*permute(partsum_psi_r_times_re_r_hat_backward,[3,2,1])-imag(permute(h_hat(:,:,Kp+1:K,t),[1,3,2])).*permute(partsum_psi_r_times_im_r_hat_backward,[3,2,1]))./nu_x(:,:,:,t));
            im_y_tilde_backward(:,:,:,t) = cat(2,alpha(t)*ones(N,Kp,M),beta(t)*ones(N,Kd,M)).*(imag(permute(x_hat(:,:,:,t),[3,2,1])).*permute(partsum_psi_q_times_re_q_hat_backward,[1,3,2])+real(permute(x_hat(:,:,:,t),[3,2,1])).*permute(partsum_psi_q_times_im_q_hat_backward,[1,3,2]))./nu_h(:,:,:,t)...
                                           +cat(2,zeros(N,Kp,M),(imag(permute(h_hat(:,:,Kp+1:K,t),[1,3,2])).*permute(partsum_psi_r_times_re_r_hat_backward,[3,2,1])+real(permute(h_hat(:,:,Kp+1:K,t),[1,3,2])).*permute(partsum_psi_r_times_im_r_hat_backward,[3,2,1]))./nu_x(:,:,:,t));

            nu_x_backward(:,:,:,t) = (-(real(permute(h_hat(:,:,Kp+1:K,t),[1,3,2])).*real(y_tilde(:,Kp+1:K,:,t))+imag(permute(h_hat(:,:,Kp+1:K,t),[1,3,2])).*imag(y_tilde(:,Kp+1:K,:,t))).*permute(partsum_psi_r_times_re_r_hat_backward,[3,2,1])...
                                      -(real(permute(h_hat(:,:,Kp+1:K,t),[1,3,2])).*imag(y_tilde(:,Kp+1:K,:,t))-imag(permute(h_hat(:,:,Kp+1:K,t),[1,3,2])).*real(y_tilde(:,Kp+1:K,:,t))).*permute(partsum_psi_r_times_im_r_hat_backward,[3,2,1])...
                                      +abs(permute(h_hat(:,:,Kp+1:K,t),[1,3,2])).^2.*permute(partsum_psi_r_square_times_psi_r_backward,[3,2,1]))./nu_x(:,:,:,t).^2;

            nu_y_backward(:,:,:,t) = nu_h_backward(:,:,:,t) + cat(2,zeros(N,Kp,M),nu_x_backward(:,:,:,t));
            
            %%%%%%%%%%
            partsum_re_y_tilde_backward = repmat(sum(re_y_tilde_backward(:,:,:,t),3),1,1,M) - re_y_tilde_backward(:,:,:,t);
            partsum_im_y_tilde_backward = repmat(sum(im_y_tilde_backward(:,:,:,t),3),1,1,M) - im_y_tilde_backward(:,:,:,t);
            partsum_nu_y_backward       = repmat(sum(nu_y_backward(:,:,:,t)      ,3),1,1,M) - nu_y_backward(:,:,:,t);
            %%%%%%%%%%
            
            re_x_hat_backward(:,:,:,t) = cat(2,zeros(M,Kp,N),(1-permute(repmat(zeta_x(:,t),1,Kd,M),[3,2,1])).*re_x_hat_backward(:,Kp+1:K,:,t+1))...
                                         -real(permute(h_hat(:,:,:,t),[2,3,1])).*permute(partsum_re_y_tilde_backward,[3,2,1])-imag(permute(h_hat(:,:,:,t),[2,3,1])).*permute(partsum_im_y_tilde_backward,[3,2,1])...
                                         +2*real(x_hat(:,:,:,t)).*(permute(psi_h(:,:,:,t),[2,3,1]).*permute(partsum_nu_y_backward,[3,2,1])-cat(2,alpha(t)*ones(M,Kp,N),beta(t)*ones(M,Kd,N))./permute(nu_h(:,:,:,t),[3,2,1]).*permute(partsum_psi_q_square_times_psi_q_backward,[2,3,1]))...
                                         +cat(2,alpha(t)*ones(M,Kp,N),beta(t)*ones(M,Kd,N)).*(real(permute(y_tilde(:,:,:,t),[3,2,1])).*permute(partsum_psi_q_times_re_q_hat_backward,[2,3,1])+imag(permute(y_tilde(:,:,:,t),[3,2,1])).*permute(partsum_psi_q_times_im_q_hat_backward,[2,3,1]))./permute(nu_h(:,:,:,t),[3,2,1]);
            im_x_hat_backward(:,:,:,t) = cat(2,zeros(M,Kp,N),(1-permute(repmat(zeta_x(:,t),1,Kd,M),[3,2,1])).*im_x_hat_backward(:,Kp+1:K,:,t+1))...
                                         +imag(permute(h_hat(:,:,:,t),[2,3,1])).*permute(partsum_re_y_tilde_backward,[3,2,1])-real(permute(h_hat(:,:,:,t),[2,3,1])).*permute(partsum_im_y_tilde_backward,[3,2,1])...
                                         +2*imag(x_hat(:,:,:,t)).*(permute(psi_h(:,:,:,t),[2,3,1]).*permute(partsum_nu_y_backward,[3,2,1])-cat(2,alpha(t)*ones(M,Kp,N),beta(t)*ones(M,Kd,N))./permute(nu_h(:,:,:,t),[3,2,1]).*permute(partsum_psi_q_square_times_psi_q_backward,[2,3,1]))...
                                         +cat(2,alpha(t)*ones(M,Kp,N),beta(t)*ones(M,Kd,N)).*(imag(permute(y_tilde(:,:,:,t),[3,2,1])).*permute(partsum_psi_q_times_re_q_hat_backward,[2,3,1])-real(permute(y_tilde(:,:,:,t),[3,2,1])).*permute(partsum_psi_q_times_im_q_hat_backward,[2,3,1]))./permute(nu_h(:,:,:,t),[3,2,1]);
            
            psi_x_backward(:,:,:,t) = (abs(permute(h_hat(:,:,:,t),[2,3,1])).^2+permute(psi_h(:,:,:,t),[2,3,1])).*permute(partsum_nu_y_backward,[3,2,1])+Phi*permute(nu_h_backward(:,:,:,t),[3,2,1])...
                                      +cat(2,zeros(M,Kp,N),(1-permute(repmat(zeta_x(:,t),1,Kd,M),[3,2,1])).*psi_x_backward(:,Kp+1:K,:,t+1));
 
            re_h_hat_backward(:,:,:,t) = -real(permute(x_hat(:,:,:,t),[3,1,2])).*permute(partsum_re_y_tilde_backward,[1,3,2])-imag(permute(x_hat(:,:,:,t),[3,1,2])).*permute(partsum_im_y_tilde_backward,[1,3,2])...
                                         +2*real(h_hat(:,:,:,t)).*permute(psi_x(:,:,:,t),[3,1,2]).*permute(partsum_nu_y_backward,[1,3,2])+(1-repmat(zeta_h(:,t),1,M,K)).*re_h_hat_backward(:,:,:,t+1)...
                                         +cat(3,zeros(N,M,Kp),(real(permute(y_tilde(:,Kp+1:K,:,t),[1,3,2])).*permute(partsum_psi_r_times_re_r_hat_backward,[3,1,2])...
                                                              +imag(permute(y_tilde(:,Kp+1:K,:,t),[1,3,2])).*permute(partsum_psi_r_times_im_r_hat_backward,[3,1,2])...
                                                             -2*real(h_hat(:,:,Kp+1:K,t)).*permute(partsum_psi_r_square_times_psi_r_backward,[3,1,2]))./permute(nu_x(:,:,:,t),[1,3,2]));
            im_h_hat_backward(:,:,:,t) =  imag(permute(x_hat(:,:,:,t),[3,1,2])).*permute(partsum_re_y_tilde_backward,[1,3,2])-real(permute(x_hat(:,:,:,t),[3,1,2])).*permute(partsum_im_y_tilde_backward,[1,3,2])...
                                         +2*imag(h_hat(:,:,:,t)).*permute(psi_x(:,:,:,t),[3,1,2]).*permute(partsum_nu_y_backward,[1,3,2])+(1-repmat(zeta_h(:,t),1,M,K)).*im_h_hat_backward(:,:,:,t+1)...
                                         +cat(3,zeros(N,M,Kp),(imag(permute(y_tilde(:,Kp+1:K,:,t),[1,3,2])).*permute(partsum_psi_r_times_re_r_hat_backward,[3,1,2])...
                                                              -real(permute(y_tilde(:,Kp+1:K,:,t),[1,3,2])).*permute(partsum_psi_r_times_im_r_hat_backward,[3,1,2])...
                                                             -2*imag(h_hat(:,:,Kp+1:K,t)).*permute(partsum_psi_r_square_times_psi_r_backward,[3,1,2]))./permute(nu_x(:,:,:,t),[1,3,2]));
                                  
            psi_h_backward(:,:,:,t) = (abs(permute(x_hat(:,:,:,t),[3,1,2])).^2+permute(psi_x(:,:,:,t),[3,1,2])).*permute(partsum_nu_y_backward,[1,3,2])+(1-repmat(zeta_h(:,t),1,M,K)).*psi_h_backward(:,:,:,t+1)...
                                      +cat(3,zeros(N,M,Kp),Es*permute(nu_x_backward(:,:,:,t),[1,3,2]));
            
            end

        end

    end
    
    
    
    
   %% Parameter update
       
    eta_x = eta_x - STEP_SIZE .* eta_x_backward;
    eta_h = eta_h - STEP_SIZE .* eta_h_backward;
    alpha = alpha - STEP_SIZE .* alpha_backward;
    beta  = beta  - STEP_SIZE .* beta_backward;
    gamma = gamma - STEP_SIZE .* gamma_backward;
        
    zeta_x = (1+exp(-a.*eta_x)).^(-1);
    zeta_h = (1+exp(-a.*eta_h)).^(-1);
        
    alpha = max(0,alpha); % Update as output of ramp function
    beta  = max(0,beta ); % Update as output of ramp function
    gamma = max(0,gamma); % Update as output of ramp function
    
    filename = ['ExpoDecay_epoch_', num2str(epoch),'_minibatch_',num2str(minibatch),'_batchsize_',num2str(BATCH_SIZE),'.mat'];
    save(filename,'zeta_x','zeta_h','eta_x','eta_h','alpha','beta','gamma','alpha_backward','beta_backward','gamma_backward','eta_x_backward','eta_h_backward','Loss','SNRdB')
    
    Losslist = [Losslist Loss];
    
    save('Losslist.mat','Losslist')
    
    semilogy(Losslist)
    grid on
    box on
    axis([0 3000 0.1 10])
    xlabel('Iterations (Number of mini batches)')
    ylabel('Loss')
    
end

    init_minibatch = 1;
    
end

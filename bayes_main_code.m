%function bayes_main_code

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% say hello
pause(0.1)
disp('Hello.  Things have started.')
pause(0.1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define save name and set random seed
saveName=[date,'_model_solution'];
rng(now);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number of draws to perform
NN_burn=10000; % 10,000 warm-up draws
NN_post=10000; % 10,000 post-warm-up draws
thin_period=10; % thin chains keeping 1 of 10
NN_burn_thin=NN_burn/thin_period; % Total number of burn-in to keep
NN_post_thin=NN_post/thin_period; % Total number of post-burn-in to keep
NN=NN_burn+NN_post; % Total number of draws to take 
NN_thin=NN_burn_thin+NN_post_thin;% Total number of draws to keep
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load data described in manuscript
load costa_etal_2023_data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define matrices
% data error variances
DeltaMat=diag(delta2);
EMat=diag(epsilon2);
invDeltaMat=inv(DeltaMat);
invEMat=inv(EMat);
% ones and identities
I_N=eye(N);
I_K=eye(K);
I_M=eye(M);
I_NK=eye(N+K);
ONE_N=ones(N,1);
ONE_K=ones(K,1);
ONE_M=ones(M,1);
ONE_NK=ones(N+K,1);
% selection matrices
FMat=zeros(N,N+K);
GMat=zeros(M,K);
HMat=zeros(M,N+K);
for nn=1:N
    FMat(nn,nn)=1;
end
for mm=1:M
    GMat(mm,m2k(mm))=1;
end
for mm=1:M
    HMat(mm,N+m2k(mm))=1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% make locations noisy to avoid off-diagonal zeros in distance matrices
latitude=latitude+1e-4*randn(size(latitude));
longitude=longitude+1e-4*randn(size(longitude));
depth=depth+1e-2*randn(size(depth));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define distance matrix
% if using latitude longitude
D=EarthDistances([longitude latitude]);
% if using depth
%D=abs(depth-depth');
% ensure symmetry
D=0.5*(D+D');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% locations for posterior prediction
minLon=round(min(longitude));
maxLon=round(max(longitude));
minLat=round(min(latitude));
maxLat=round(max(latitude));
postLon=minLon:2:maxLon; % regular 2-degree grid
postLat=minLat:2:maxLat; % regular 2-degree grid
[postLon postLat]=meshgrid(postLon,postLat);
postLon=reshape(postLon,numel(postLon),1);
postLat=reshape(postLat,numel(postLat),1);
NPOST=numel(postLat);
ONE_NPOST=ones(NPOST,1);
DTOT=EarthDistances([ [longitude; postLon] [latitude; postLat] ]);
SPOST=zeros(NN_post/thin_period,NPOST);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define priors and hyperparameters
RD=round(D);
logx=log(x);
ysurf=y(find(z<10));
eta_mu=mean(logx);
zeta2_mu=4^2*(std(logx))^2;
eta_phi=0.5*(max(log(1./RD(RD~=0)))+min(log(1./RD(RD~=0))));
zeta2_phi=4^2*(0.25*(min(log(1./RD(RD~=0)))-max(log(1./RD(RD~=0)))))^2;
eta_rho=0.5*(max(log(1./RD(RD~=0)))+min(log(1./RD(RD~=0))));
zeta2_rho=4^2*(0.25*(min(log(1./RD(RD~=0)))-max(log(1./RD(RD~=0)))))^2;
eta_nu=mean(ysurf);
zeta2_nu=4^2*var(ysurf);
xi_gamma=1/2;
chi_gamma=1/2*var(logx);
xi_sigma=1/2;
chi_sigma=1/2*var(logx);
xi_tau=1/2;
chi_tau=1/2*var(ysurf);
xi_pi=1/2;
chi_pi=1/2*var(ysurf);
xi_kappa=1/2;
chi_kappa=1/2*var(y);
clear logx ysurf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set initial values
mu=eta_mu+sqrt(zeta2_mu)*randn(1);
nu=eta_nu+sqrt(zeta2_nu)*randn(1);
phi=eta_phi+sqrt(zeta2_phi)*randn(1);
rho=eta_rho+sqrt(zeta2_rho)*randn(1);
gamma2=min([10 1/randraw('gamma', [0,1/chi_gamma,xi_gamma], [1,1])]); % use min to prevent needlessly large values
sigma2=min([10 1/randraw('gamma', [0,1/chi_sigma,xi_sigma], [1,1])]); % use min to prevent needlessly large values
tau2=min([50 1/randraw('gamma', [0,1/chi_tau,xi_tau], [1,1])]); % use min to prevent needlessly large values
pi2=min([50 1/randraw('gamma', [0,1/chi_pi,xi_pi], [1,1])]); % use min to prevent needlessly large values
kappa2=min([50 1/randraw('gamma', [0,1/chi_kappa,xi_kappa], [1,1])]); % use min to prevent needlessly large values

SigmaMat=sigma2*exp(-exp(phi)*D);
SigmaMat=0.5*(SigmaMat+SigmaMat');
invSigmaMat=inv(SigmaMat);
CMat=exp(-exp(phi)*(D));
CMat=0.5*(CMat+CMat');
invCMat=inv(CMat);

PiMat=pi2*exp(-exp(rho)*D((N+1):(N+K),(N+1):(N+K)));
PiMat=0.5*(PiMat+PiMat');
invPiMat=inv(PiMat);
PMat=exp(-exp(rho)*D((N+1):(N+K),(N+1):(N+K)));
PMat=0.5*(PMat+PMat');
invPMat=inv(PMat);

s=mvnrnd(mu*ONE_NK,SigmaMat)';
r=mvnrnd(s,gamma2*I_NK)';
beta=mvnrnd(nu*ONE_K,PiMat)';
alpha=mvnrnd(beta,tau2*I_K)';
d=z;
t=y;
Xi=diag(GMat*alpha)*exp(-1000*lambda230*diag(d)*exp(-HMat*r));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% allocate space
MU=zeros(NN,1);
NU=zeros(NN,1);
PHI=zeros(NN,1);
RHO=zeros(NN,1);
GAMMA2=zeros(NN,1);
TAU2=zeros(NN,1);
KAPPA2=zeros(NN,1);
SIGMA2=zeros(NN,1);
PI2=zeros(NN,1);

S=zeros(NN,N+K);
R=zeros(NN,N+K);
DEE=zeros(NN,M);
TEE=zeros(NN,M);
ALPHA=zeros(NN,K);
BETA=zeros(NN,K);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN LOOP
% loop through sampler
for nn=1:NN, disp(num2str(nn))
    tic

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample from mu
    v=[]; psi=[];
    v=eta_mu/zeta2_mu+ONE_NK'*invSigmaMat*s;
    psi=inv(ONE_NK'*invSigmaMat*ONE_NK+1/zeta2_mu);
    sample=[]; sample=psi*v+sqrt(psi)*randn(1);
    mu=sample;
    MU(nn)=mu;
    clear v psi sample
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample from nu
    v=[]; psi=[];
    v=eta_nu/zeta2_nu+ONE_K'*invPiMat*beta;
    psi=inv(ONE_K'*invPiMat*ONE_K+1/zeta2_nu);
    sample=[]; sample=psi*v+sqrt(psi)*randn(1);
    nu=sample;
    NU(nn)=nu;
    clear v psi sample
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample from phi
    % metropolis sampler
    Phi_now=phi;
    Phi_std=0.5;
    Phi_prp=Phi_now+Phi_std*randn(1);
    SigmaMat_now=sigma2*exp(-exp(Phi_now)*D);
    SigmaMat_prp=sigma2*exp(-exp(Phi_prp)*D);
    invSigmaMat_now=inv(SigmaMat_now);
    invSigmaMat_prp=inv(SigmaMat_prp);
 	ins_now=-0.5*(s-mu*ONE_NK)'*invSigmaMat_now*(s-mu*ONE_NK)-0.5*((Phi_now-eta_phi)^2)/zeta2_phi;
   	ins_prp=-0.5*(s-mu*ONE_NK)'*invSigmaMat_prp*(s-mu*ONE_NK)-0.5*((Phi_prp-eta_phi)^2)/zeta2_phi;
  	MetFrac=det(SigmaMat_prp*invSigmaMat_now)^(-1/2)*exp(ins_prp-ins_now);
   	success_rate=min(1,MetFrac);
   	if rand(1)<=success_rate
     	Phi_now=Phi_prp; 
    end
  	phi=Phi_now;
    PHI(nn)=phi;
    % redefine relevant matrices
    SigmaMat=sigma2*exp(-exp(phi)*D);
    SigmaMat=0.5*(SigmaMat+SigmaMat');
    invSigmaMat=inv(SigmaMat);
    CMat=exp(-exp(phi)*D);
    CMat=0.5*(CMat+CMat');
    invCMat=inv(CMat);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample from phi
    % metropolis sampler
    Rho_now=rho;
    Rho_std=0.5;
    Rho_prp=Rho_now+Rho_std*randn(1);
    PiMat_now=pi2*exp(-exp(Rho_now)*D((N+1):(N+K),(N+1):(N+K)));
    PiMat_prp=pi2*exp(-exp(Rho_prp)*D((N+1):(N+K),(N+1):(N+K)));
    invPiMat_now=inv(PiMat_now);
    invPiMat_prp=inv(PiMat_prp);
 	ins_now=-0.5*(beta-nu*ONE_K)'*invPiMat_now*(beta-nu*ONE_K)-0.5*((Rho_now-eta_rho)^2)/zeta2_rho;
   	ins_prp=-0.5*(beta-nu*ONE_K)'*invPiMat_prp*(beta-nu*ONE_K)-0.5*((Rho_prp-eta_rho)^2)/zeta2_rho;
  	MetFrac=det(PiMat_prp*invPiMat_now)^(-1/2)*exp(ins_prp-ins_now);
   	success_rate=min(1,MetFrac);
   	if rand(1)<=success_rate
     	Rho_now=Rho_prp; 
    end
  	rho=Rho_now;
    RHO(nn)=rho;
    % redefine relevant matrices
    PiMat=pi2*exp(-exp(rho)*D((N+1):(N+K),(N+1):(N+K)));
    PiMat=0.5*(PiMat+PiMat');
    invPiMat=inv(PiMat);
    PMat=exp(-exp(rho)*D((N+1):(N+K),(N+1):(N+K)));
    PMat=0.5*(PMat+PMat');
    invPMat=inv(PMat);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample from gamma2
    gamma2=1/randraw('gamma', [0,1/(chi_gamma+1/2*(r-s)'*(r-s)),...
     	(xi_gamma+(N+K)/2)], [1,1]);
    GAMMA2(nn)=gamma2;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample from sigma2
    sigma2=1/randraw('gamma', [0,1/(chi_sigma+1/2*(s-mu*ONE_NK)'*invCMat*(s-mu*ONE_NK)),(xi_sigma+(N+K)/2)], [1,1]);
    SIGMA2(nn)=sigma2;
    % redefine relevant matrices
    SigmaMat=sigma2*exp(-exp(phi)*D);
    SigmaMat=0.5*(SigmaMat+SigmaMat');
    invSigmaMat=inv(SigmaMat);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample from pi2
    pi2=1/randraw('gamma', [0,1/(chi_pi+1/2*(beta-nu*ONE_K)'*invPMat*(beta-nu*ONE_K)),(xi_pi+K/2)], [1,1]);
    PI2(nn)=pi2;
    % redefine relevant matrices
    PiMat=pi2*exp(-exp(rho)*D((N+1):(N+K),(N+1):(N+K)));
    PiMat=0.5*(PiMat+PiMat');
    invPiMat=inv(PiMat);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample from kappa2
    kappa2=1/randraw('gamma', [0,1/(chi_kappa+1/2*(t-Xi)'*(t-Xi)),(xi_gamma+M/2)], [1,1]);
    KAPPA2(nn)=kappa2;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample from tau2
    tau2=1/randraw('gamma', [0,1/(chi_tau+1/2*(alpha-beta)'*(alpha-beta)),(xi_tau+K/2)], [1,1]);
    TAU2(nn)=tau2;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample from s
    v=[]; psi=[]; sample=[];
    v=r/gamma2+mu*invSigmaMat*ONE_NK;
    psi=inv(1/gamma2*I_NK+invSigmaMat);
    sample=mvnrnd(psi*v,psi)';
    s=sample;
    S(nn,:)=s;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample from beta
    v=[]; psi=[]; sample=[];
    v=alpha/tau2+nu*invPiMat*ONE_K;
    psi=inv(1/tau2*I_K+invPiMat);
    sample=mvnrnd(psi*v,psi)';
    beta=sample;
    BETA(nn,:)=beta;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample from t
    v=[]; psi=[]; sample=[];
    v=invEMat*y+Xi/kappa2;
    psi=inv(invEMat+I_M/kappa2);
    sample=mvnrnd(psi*v,psi)';
    t=sample;
    TEE(nn,:)=t;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample from r
    % metropolis sampler
    for kk=1:(N+K)
        r_now=r(kk);
        R_now=r; 
        R_now(kk)=r_now;
        r_std=0.5;
        r_prp=r_now+r_std*randn(1);
        R_prp=r;
        R_prp(kk)=r_prp;
        Xi_now=diag(GMat*alpha)*exp(-1000*lambda230*diag(d)*exp(-HMat*R_now));
        Xi_prp=diag(GMat*alpha)*exp(-1000*lambda230*diag(d)*exp(-HMat*R_prp));

        if kk<=N % you're at one of the places that you have a rate NOT based on thorium from the literature
            ins_now=-0.5/delta2(kk)*(x(kk)-exp(r_now))^2-0.5/gamma2*(r_now-s(kk))^2;
            ins_prp=-0.5/delta2(kk)*(x(kk)-exp(r_prp))^2-0.5/gamma2*(r_prp-s(kk))^2;
        else % kk>N you're at one of the places with thorium cores
            iii=[]; iii=find(m2k==(kk-N));
            ins_now=-0.5/gamma2*(r_now-s(kk))^2-0.5/kappa2*(t(iii)-Xi_now(iii))'*(t(iii)-Xi_now(iii));
            ins_prp=-0.5/gamma2*(r_prp-s(kk))^2-0.5/kappa2*(t(iii)-Xi_prp(iii))'*(t(iii)-Xi_prp(iii));
        end
        metFrac=exp(ins_prp-ins_now);
    	success_rate=min(1,metFrac);
        if rand(1)<=success_rate
            r_now=r_prp;
        end
        r(kk)=r_now;
        R(nn,kk)=r(kk);
    end
    Xi=diag(GMat*alpha)*exp(-1000*lambda230*diag(d)*exp(-HMat*r));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample from alpha
    % metropolis sampler
    for kk=1:K
        alpha_now=alpha(kk);
        Alpha_now=alpha; 
        Alpha_now(kk)=alpha_now;
        alpha_std=0.5;
        alpha_prp=alpha_now+alpha_std*randn(1);
        Alpha_prp=alpha;
        Alpha_prp(kk)=alpha_prp;
        Xi_now=diag(GMat*Alpha_now)*exp(-1000*lambda230*diag(d)*exp(-HMat*r));
        Xi_prp=diag(GMat*Alpha_prp)*exp(-1000*lambda230*diag(d)*exp(-HMat*r));

        iii=[]; iii=find(m2k==(kk));
        ins_now=-0.5/tau2*(alpha_now-beta(kk))^2-0.5/kappa2*(t(iii)-Xi_now(iii))'*(t(iii)-Xi_now(iii));
        ins_prp=-0.5/tau2*(alpha_prp-beta(kk))^2-0.5/kappa2*(t(iii)-Xi_prp(iii))'*(t(iii)-Xi_prp(iii));
        metFrac=exp(ins_prp-ins_now);
    	success_rate=min(1,metFrac);
        if rand(1)<=success_rate
            alpha_now=alpha_prp;
        end
        alpha(kk)=alpha_now;
        ALPHA(nn,kk)=alpha(kk);
    end
    Xi=diag(GMat*alpha)*exp(-1000*lambda230*diag(d)*exp(-HMat*r));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sample from d
    % metropolis sampler
    for kk=1:M
        d_now=d(kk);
        D_now=d; 
        D_now(kk)=d_now;
        d_std=0.5;
        d_prp=max([d_now+d_std*randn(1) 0]);
        D_prp=d;
        D_prp(kk)=d_prp;
        Xi_now=diag(GMat*alpha)*exp(-1000*lambda230*diag(D_now)*exp(-HMat*r));
        Xi_prp=diag(GMat*alpha)*exp(-1000*lambda230*diag(D_prp)*exp(-HMat*r));
        ins_now=-0.5/kappa2*(t(kk)-Xi_now(kk));
        ins_prp=-0.5/kappa2*(t(kk)-Xi_prp(kk));
        metFrac=exp(ins_prp-ins_now);
    	success_rate=min(1,metFrac);
        if rand(1)<=success_rate&&d_prp>=ztop(kk)&&d_prp<=zbot(kk)
            d_now=d_prp;
        end
        d(kk)=d_now;
        DEE(nn,kk)=d(kk);
    end
    Xi=diag(GMat*alpha)*exp(-1000*lambda230*diag(d)*exp(-HMat*r));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    toc
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% delete burn in
KAPPA2(1:NN_burn)=[];
TAU2(1:NN_burn)=[];
GAMMA2(1:NN_burn)=[];
SIGMA2(1:NN_burn)=[];
PI2(1:NN_burn)=[];
MU(1:NN_burn)=[];
NU(1:NN_burn)=[];
PHI(1:NN_burn)=[];
RHO(1:NN_burn)=[];
S(1:NN_burn,:)=[];
R(1:NN_burn,:)=[];
DEE(1:NN_burn,:)=[];
TEE(1:NN_burn,:)=[];
ALPHA(1:NN_burn,:)=[];
BETA(1:NN_burn,:)=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% thin
GAMMA2=GAMMA2(1:thin_period:NN_post);
SIGMA2=SIGMA2(1:thin_period:NN_post);
TAU2=TAU2(1:thin_period:NN_post);
KAPPA2=KAPPA2(1:thin_period:NN_post);
PI2=PI2(1:thin_period:NN_post);
MU=MU(1:thin_period:NN_post);
NU=NU(1:thin_period:NN_post);
PHI=PHI(1:thin_period:NN_post);
RHO=RHO(1:thin_period:NN_post);
S=S(1:thin_period:NN_post,:);
R=R(1:thin_period:NN_post,:);
DEE=DEE(1:thin_period:NN_post,:);
TEE=TEE(1:thin_period:NN_post,:);
ALPHA=ALPHA(1:thin_period:NN_post,:);
BETA=BETA(1:thin_period:NN_post,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars -except saveName GAMMA2 SIGMA2 TAU2 KAPPA2 MU NU ALPHA PHI S R DEE TEE longitude latitude depth N K M I_* ONE_* min* max* post* *POST* DTOT core depth delta2 epsilon2 z* lambda230 m2k reference type x y FMat GMat HMat PI2 RHO BETA

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% perform posterior prediction
for nn=1:numel(NU), disp(num2str(nn))
    A=[]; A=SIGMA2(nn)*exp(-exp(PHI(nn))*DTOT(((N+K+1):(N+K+NPOST)),((N+K+1):(N+K+NPOST))));
    B=[]; B=SIGMA2(nn)*exp(-exp(PHI(nn))*DTOT(1:(N+K),1:(N+K)))+GAMMA2(nn)*I_NK;
    C=[]; C=SIGMA2(nn)*exp(-exp(PHI(nn))*DTOT(( (N+K+1):(N+K+NPOST) ),( 1:(N+K) )));
    yy=R(nn,:)';
    muy=MU(nn)*ONE_NK;
    mux=MU(nn)*ONE_NPOST;
    MN=mux+C*inv(B)*(yy-muy);
    CV=A-C*inv(B)*C'; CV=0.5*(CV+CV');
    sample=mvnrnd(MN,CV);
    SPOST(nn,:)=sample;
    clear A B C yy muy mux sample MN CV
end
postLon=minLon:2:maxLon;
postLat=minLat:2:maxLat;
SPOSTreshape=reshape(SPOST,numel(SIGMA2),numel(postLat),numel(postLon));
[postLon postLat]=meshgrid(postLon,postLat);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% save
save(saveName,'GAMMA2','SIGMA2','TAU2','KAPPA2','MU','NU','ALPHA','PHI','S','R','DEE','TEE','longitude','latitude','depth','N','K','M','I_*','ONE_*','min*','max*','post*','*POST*','DTOT','core','depth','delta2','epsilon2','z*','lambda230','m2k','reference','type','x','y','FMat','GMat','HMat','PI2','RHO','BETA','*POST*','*post*')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
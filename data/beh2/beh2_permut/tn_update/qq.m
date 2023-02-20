clear all

%Q - number of qubits, should be even
Q = 6;
%SL - slow down factor, increase if the optimization becomes unstable
SL = 1;

%load Hamiltonian from a file
load('ham_full.mat','H')
d = 2^(Q/2);
H = reshape(H,d,d,d,d);

%full state |psi> is a product: |psi> = |psiL> \otimes |psiR>

%start with random, normalized states
psiL = rand(2^(Q/2),1)-0.5 + 1.i*(rand(2^(Q/2),1)-0.5);
psiL = psiL/norm(psiL);
psiR = rand(2^(Q/2),1)-0.5 + 1.i*(rand(2^(Q/2),1)-0.5);
psiR = psiR/norm(psiR);

psiL

%update until convergence is reached or
%number of iterations exceeds maximum
en0 = 0;en1 = 1;it = 0;
while abs(en0-en1) > 1.e-10 & it < 1000
    it = it+1;

    %update |psiL>
    envL = scon({psiR,H,conj(psiL),conj(psiR)},...
    {[1 -2],[-1 1 2 3],[2 -3],[3 -4]});
    psiL = conj(envL) - SL*psiL;
    psiL = psiL/norm(psiL);

    %update |psiR>
    envR = scon({psiL,H,conj(psiL),conj(psiR)},...
    {[1 -2],[1 -1 2 3],[2 -3],[3 -4]});
    psiR = conj(envR) - SL*psiR;
    psiR = psiR/norm(psiR);

    en0 = en1;
    en1 = energy(psiL,psiR,H);
    disp(num2str(en1,13))

end

%
